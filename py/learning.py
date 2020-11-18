import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mido import MidiFile
import os

# Midi should have octave in integers [0,10] (so eleven octaves)
# Returns (octave,pitchClass)
def midiNoteToRepr(midiNote):
  return divmod(midiNote,12)

fockSize = 10
# 12 pitch classes, and 11 octaves
singleNoteSize = 12 + 11
totalFockSize = 0
fockOffsets = []
for i in range(fockSize):
  fockOffsets.append(totalFockSize)
  totalFockSize += (i + 1) * singleNoteSize
fockOffsets.append(totalFockSize)

# Add in the time density
maxTimeDensity = 200
useCategoricalTime = False
if useCategoricalTime:
  timeDensitySize = maxTimeDensity // 4
  noteSpecSize = timeDensitySize + totalFockSize
else:
  noteSpecSize = 1 + totalFockSize

print("Total fock size is {}".format(totalFockSize))
print("Fock offsets are {}".format(fockOffsets))

def midiNoteToSingleNoteChunk(midiNote):
  (octave,pitchClass) = divmod(midiNote,12)
  vec = np.zeros(singleNoteSize)
  vec[pitchClass] = 1
  vec[12 + octave] = 1
  return vec

# Build a NoteSpec out of the current midi situation during this particular epoch
def mkNoteSpec(heldNotes,decayingNotes,timeDensity):
  allNotes = heldNotes + decayingNotes
  fockVec = np.zeros(totalFockSize)
  if len(allNotes) > fockSize:
    #print("!!! Fock size of {} exceeded by {} simultaneous notes; dropping {} notes !!!".format(fockSize,len(allNotes), len(allNotes) - fockSize))
    allNotes = allNotes[:fockSize]
  startIdx = fockOffsets[len(allNotes) - 1]
  for i in range(len(allNotes)):
    fockVec[startIdx + i * singleNoteSize : startIdx + (i + 1) * singleNoteSize] = midiNoteToSingleNoteChunk(allNotes[i])
  if useCategoricalTime:
    timeVec = np.zeros(timeDensitySize)
    timeVec[(timeDensity // 4) - 1] = 1
    return np.insert(fockVec,0,timeVec)
  else:
    return np.insert(fockVec,0,timeDensity)

def getTracksFromMidi(filename,verbose=False):
  mid = MidiFile(filename)
  tracks = []
  minTrackSize = 200
  for i, track in enumerate(mid.tracks):
    if verbose:
      print('Track {}: {}'.format(i, track.name))
    heldNotes = []
    toUnHold = []
    lastTime = 0
    noteSpecs = []
    for msg in track:
      if msg.time != 0:
        noteSpec = mkNoteSpec(heldNotes,toUnHold,min(lastTime,maxTimeDensity))
        noteSpecs.append(noteSpec)
        for n in toUnHold:
          if n in heldNotes:
            heldNotes.remove(n)
        toUnHold = []
        lastTime = msg.time
      if msg.type == 'note_on':
        #print("Note",msg.note,"on with time=",msg.time)
        if msg.note not in heldNotes:
          heldNotes.append(msg.note)
      elif msg.type == 'note_off':
        #print("Note",msg.note,"off with time=",msg.time)
        if msg.note in heldNotes:
          toUnHold.append(msg.note)
      else:
        #print(msg)
        x=1
    if verbose:
      print("Found {} Note Specs".format(len(noteSpecs)))
    if(len(noteSpecs) >= minTrackSize):
      tracks.append(np.stack(noteSpecs,axis=0))
    elif verbose:
      print("Track Discarded XXX")
  return tracks

def mkDatasetFromTrack(noteSequence):
  # The maximum length sentence you want for a single input in characters
  seq_length = 20
  examples_per_epoch = len(noteSequence)//(seq_length+1)

  # Create training examples / targets
  sequences = tf.data.Dataset.from_tensor_slices(noteSequence).batch(seq_length+1, drop_remainder=True)

  def split_input_target(chunk):
    inputNoteSeq = chunk[:-1]
    targetNoteSeq = chunk[1:]
    return inputNoteSeq, targetNoteSeq

  dataset = sequences.map(split_input_target)

  # Batch size
  BATCH_SIZE = 10

  # Buffer size to shuffle the dataset
  # (TF data is designed to work with possibly infinite sequences,
  # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
  # it maintains a buffer in which it shuffles elements).
  BUFFER_SIZE = 100

  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
  return dataset

def fileToData(filename):
  tracks = getTracksFromMidi(filename)
  dataset = None
  for track in tracks:
    if dataset is None:
      dataset = mkDatasetFromTrack(track)
    else:
      dataset.concatenate(mkDatasetFromTrack(track))
  return dataset

max_files = 5
dataset = None
for filename in os.listdir("midis"):
  max_files -= 1
  if max_files < 0:
    break
  fullPath = os.path.join("midis",filename)
  print("Using",fullPath)
  try:
    fileSet = fileToData(fullPath)
    if dataset is None:
      dataset = fileSet
    else:
      dataset.concatenate(fileSet)
  except ValueError as err:
    print("!!! Error dealing with midi file:",fullPath)
    print("!!!",err)

dataset = dataset.shuffle(100)
print("Dataset cardinality:",dataset.cardinality().numpy())



#if useCategoricalTime:
#  print(tracks[1][:,:timeDensitySize])
#else:
#  print(tracks[1][:,0])

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(None,noteSpecSize)))
model.add(layers.Bidirectional(layers.LSTM(512,return_sequences=True)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(noteSpecSize,activation = 'relu'))

model.summary()

def postProcess(timeAndLogits):
  if useCategoricalTime:
    return tf.nn.softmax(timeAndLogits).numpy()
  else:
    return (timeAndLogits[0],tf.nn.softmax(timeAndLogits[1:]).numpy())

def displayBatch(batch,from_logits=False,title=""):
  print("Time densities:")
  print(batch[:,0].numpy())
  if from_logits:
    probs = tf.nn.softmax(batch[:,1:]).numpy()
  else:
    probs = batch[:,1:]
  #print(probs)
  #print(probs.shape)
  plt.scatter(range(len(probs[-1])),np.log(probs[-1] + 1))

def displayAbout(model):
  for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    #print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    print("Input:")
    displayBatch(input_example_batch[0])
    print()
    print("Predictions:")
    displayBatch(example_batch_predictions[0],from_logits=True)
    plt.title("Inputs vs predictions")
    plt.show(block=True)
    plt.title("Time density distribution comparison")
    plt.hist([input_example_batch[0][:,0],example_batch_predictions[0][:,0]],label=["Actual time density distribution","Predicted time density distribution"])
    #plt.vlines([np.mean(input_example_batch[0][:,0])],label="Mean actual time density")
    plt.legend()
    plt.show(block=True)

def lossfn(actual, pred):
  predTimeDensity = pred[:,:,0]
  actualTimeDensity = actual[:,:,0]
  alpha = 0.2
  timeLoss = (predTimeDensity - actualTimeDensity) * (predTimeDensity - actualTimeDensity)
  beta = 0.5
  timeShapeLoss = tf.keras.losses.kullback_leibler_divergence(actualTimeDensity,predTimeDensity)
  fockProbsAct = []
  fockProbsPre = []
  cce = []
  for i in range(fockSize):
    startIdx = 1 + fockOffsets[i]
    endIdx = 1 + fockOffsets[i + 1]
    fockProbsAct.append(tf.keras.backend.sum(actual[:,:,startIdx : endIdx],axis=2))
    fockProbsPre.append(tf.keras.backend.sum(pred[:,:,startIdx : endIdx],axis=2))
    cce.append(tf.keras.losses.categorical_crossentropy(actual[:,:,startIdx : endIdx], pred[:,:, startIdx : endIdx], from_logits=True))
  fockProbsActTensor = tf.keras.backend.stack(fockProbsAct,axis=2)
  fockProbsPreTensor = tf.keras.backend.stack(fockProbsPre,axis=2)
  cceTensor = tf.keras.backend.stack(cce,axis=2)
  actualFockIdx = tf.math.argmax(fockProbsActTensor,axis=2)
  # This is the entropy from correctly categorizing the number of notes to play simultaneously
  fockIndexEntropy = -tf.math.log(tf.gather(fockProbsPreTensor,actualFockIdx,batch_dims=2))

  cceEntropy = tf.gather(cceTensor,actualFockIdx,batch_dims=2)

  #cce = tf.keras.losses.categorical_crossentropy(actual[:,:,1:], pred[:,:,1:], from_logits=True)
  return alpha * timeLoss + fockIndexEntropy + cceEntropy


model.compile(optimizer='adam', loss=lossfn, metrics=['accuracy'])


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

if input("Go?") != "y":
  quit()

#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

displayAbout(model)

model.fit(dataset, epochs=10, callbacks=[checkpoint_callback])

#model.evaluate(dataset)
displayAbout(model)

def writeBatch(batch,from_logits=False,title=""):
  heldNotes = []
  for i in range(len(batch[:,0])):
    timeDensity = batch[i,0]
  #print("Time densities:")
  #print(batch[:,0].numpy())
  #if from_logits:
  #  probs = tf.nn.softmax(batch[:,1:]).numpy()
  #else:
  #  probs = batch[:,1:]
  #print(probs)
  #print(probs.shape)
  #plt.scatter(range(len(probs[0])),np.log(probs[0]))


