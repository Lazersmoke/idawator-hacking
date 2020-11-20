import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mido import MidiFile, Message, MidiTrack
from itertools import permutations
import os

np.set_printoptions(precision=2)

# This is the size of the Fock space of simulatneous notes
# The Fock space should be thought of as:
# (note) `direct sum` (note X note) `direct sum` (note X note X note) ...
# Up to the fockSize-fold cartesian product: \prod_{i=1}^fockSize note_i
fockSize = 10

# A single note in the fock representation described above
# 12 pitch classes, and 11 octaves, one-hot encoding
# 11 octaves because MIDI comes in [0,127], last octave is [121,127]
singleNoteSize = 12 + 11

# Compute some constants based on the fockSize and singleNoteSize
totalFockSize = 0
fockOffsets = []
for i in range(fockSize):
  fockOffsets.append(totalFockSize)
  totalFockSize += (i + 1) * singleNoteSize
# Include the ending offset; this is useful sometimes
fockOffsets.append(totalFockSize)

# totalFockSize is the dimension of the Fock space
print("Total fock size is {}".format(totalFockSize))

# fockOffsets holds the offsets to the different direct summands of the Fock space
# The summand with k simultaneous notes is between fockOffsets[k-1] and fockOffsets[k]
print("Fock offsets are {}".format(fockOffsets))

# Add in the time density as an extra dimension
# Time density is the amount of time that this particular chunk of
# the MIDI takes up (in midi ticks). It's a density because the non-uniform
# MIDI is being squished and stretched to be uniform (timeseries)
# so we need to keep the density of that embedding around

# NoteSpec = (Time part) `direct sum` (Fock space)
# Note that it is not one-hot encoded
noteSpecSize = 1 + totalFockSize

# Convert a midi note in [0,127] to a one-hot encoded
# vector of size singleNoteSize
def midiNoteToSingleNoteChunk(midiNote):
  (octave,pitchClass) = divmod(midiNote,12)
  vec = np.zeros(singleNoteSize)
  vec[pitchClass] = 1
  vec[12 + octave] = 1
  return vec

# Build a NoteSpec out of the current midi situation during this particular chunk of midi
def mkNoteSpec(heldNotes,decayingNotes,timeDensity):
  # We don't distinguish between notes that are being freshly played right now
  # and those which are now in the release stage of ADSR
  allNotes = heldNotes + decayingNotes
  fockVec = np.zeros(totalFockSize)
  # If there are more simultaneous notes than fockSize, we can't represent it
  # The current solution is to arbitrarily drop them by slicing the list
  # This isn't particularly stable due to heldNotes being in potentially arbitrary orders
  if len(allNotes) > fockSize:
    allNotes = allNotes[:fockSize]
    # This print statement will bottleneck your parsing in some cases
    #print("!!! Fock size of {} exceeded by {} simultaneous notes; dropping {} notes !!!".format(fockSize,len(allNotes), len(allNotes) - fockSize))

  # Decide which direct summand of the fock space we fall under
  startIdx = fockOffsets[len(allNotes) - 1]
  # Place each currently playing note into the correct slot
  for i in range(len(allNotes)):
    fockVec[startIdx + i * singleNoteSize : startIdx + (i + 1) * singleNoteSize] = midiNoteToSingleNoteChunk(allNotes[i])
  # Include the time density in front of the fock vector to make this a NoteSpec
  return np.insert(fockVec,0,timeDensity)

# Given a MIDI filename on disk, get the list of tracks
def getTracksFromMidi(filename,verbose=False):
  mid = MidiFile(filename)
  tracks = []
  for i, track in enumerate(mid.tracks):
    if verbose:
      print('Track {}: {}'.format(i, track.name))
    # The notes that are currently being played
    # This must be tracked between midi messages
    heldNotes = []
    # Notes that are turning off
    toUnHold = []
    lastTime = 0
    noteSpecs = []
    for msg in track:
      # MIDI typically issues several commands with time=0, and the last one with
      # time equal to the time until the next command in midi ticks
      # If the time isn't zero, then we have all the commands from this chunk
      # and it's time to write it out

      # On top of this, we treat anything short of 4 midi ticks as essentially happening in zero time.
      if msg.time > 3:
        # Cap the time density so that pathologically long
        # notes in the data don't destroy everything
        maxTimeDensity = 200

        if len(heldNotes) + len(toUnHold) > 0:
          noteSpec = mkNoteSpec(heldNotes,toUnHold,min(lastTime,maxTimeDensity))
          noteSpecs.append(noteSpec)

        # Formally advance the time to the next chunk
        lastTime = msg.time

        # All the notes that were turning off in the last chunk are now fully off
        for n in toUnHold:
          if n in heldNotes:
            heldNotes.remove(n)
        toUnHold = []

      # Adjust the note states based on the MIDI message
      if msg.type == 'note_on' and msg.note not in heldNotes:
        heldNotes.append(msg.note)
      elif msg.type == 'note_off' and msg.note in heldNotes:
        toUnHold.append(msg.note)

    if verbose:
      print("Found {} Note Specs".format(len(noteSpecs)))

    # Too-short tracks are typically like effect tracks
    # or other weird things that we don't want to learn on
    minTrackSize = 200
    if(len(noteSpecs) >= minTrackSize):
      tracks.append(np.stack(noteSpecs,axis=0))
    elif verbose:
      print("Track Discarded XXX")
  return tracks

# Create a tensorflow dataset of prediction examples (ie, flashcards)
# based on the given track (which is just a list of NoteSpec's)
def mkDatasetFromTrack(track):
  # The length in NoteSpec's of sequences that we should (a) learn on and (b) generate
  seq_length = 50

  # Chop up the input track into slices
  # Makes about len(track)/(seq_length + 1) examples out of the track
  sequences = tf.data.Dataset.from_tensor_slices(track).batch(seq_length+1, drop_remainder=True)

  # For each example we have:
  # input is  [n    ,    n + seq_length]
  # output is [1 + n,1 + n + seq_length]
  def split_input_target(chunk):
    inputNoteSeq = chunk[:-1]
    targetNoteSeq = chunk[1:]
    return inputNoteSeq, targetNoteSeq

  # Turn the database of slices into a database of examples
  dataset = sequences.map(split_input_target)
  return dataset

# Get the database of examples for a midi file
def fileToData(filename):
  tracks = getTracksFromMidi(filename)
  dataset = None
  for track in tracks:
    if dataset is None:
      dataset = mkDatasetFromTrack(track)
    else:
      dataset = dataset.concatenate(mkDatasetFromTrack(track))
  return dataset

# Use up to this many files from the "midis" folder
# to build the database of examples
max_files = 50
dataset = None
midiFiles = os.listdir("clean_midi")
np.random.shuffle(midiFiles)
for filename in midiFiles:
  max_files -= 1
  if max_files < 0:
    break
  fullPath = os.path.join("clean_midi",filename)
  if not os.path.isfile(fullPath):
    continue
  print("Using",fullPath)
  try:
    fileSet = fileToData(fullPath)
    if dataset is None:
      dataset = fileSet
    elif fileSet is not None:
      dataset = dataset.concatenate(fileSet)
  # Some MIDI files have data values > 127, which the mido library doesn't like
  # so it throws these errors. We catch them and ignore the culprit file
  except ValueError as err:
    print("!!! ValueError dealing with midi file:",fullPath)
    print("!!!",err)
  except OSError as err:
    print("!!! OSError dealing with midi file:",fullPath)
    print("!!!",err)
  except:
    print("!!! Other error dealing with midi file:",fullPath)

# Peel off the validation set
validationSet = dataset.shard(num_shards=2,index=1).batch(50)
dataset = dataset.shard(num_shards=2,index=0)

# Batch the dataset into chunks of 50 examples each to make training more managable
dataset = dataset.shuffle(100).batch(50, drop_remainder=True)


# Switch to use fewer than the full dataset for fast training
max_batches = 10
if max_batches > 0:
  dataset = dataset.take(max_batches)

# Our model!
model = keras.Sequential()
# Input is some sequence of NoteSpec's of unspecified length
model.add(layers.InputLayer(input_shape=(None,noteSpecSize)))
# Use a Bidirectional LSTM to remember states an all that good stuff
model.add(layers.Bidirectional(layers.LSTM(512,return_sequences=True)))
# The dropout layer prevents overfitting (via black magic)
model.add(layers.Dropout(0.2))
# A finalizing simple neural network layer using relu because relu is love relu is life
model.add(layers.Dense(noteSpecSize,activation = 'relu'))

model.summary()

# Write out a sequence of NoteSpec's to MIDI (using maximum probabilities)
# Technically this is wrong and we should sample instead, but this works OK
def writeBatch(batch,from_logits=True,title="batch"):
  mid = MidiFile()
  track = MidiTrack()
  mid.tracks.append(track)
  # Set grand piano
  track.append(Message('program_change', program=1, time=0))
  # volume up
  track.append(Message('control_change', control=7, value=127, time=0))
  # sustain pedal??
  # The dataset MIDIs do this and it seems to help
  track.append(Message('control_change', control=64, value=127, time=0))

  # This is almost exactly the inverse of the MIDI parsing stuff
  heldNotes = []
  for i in range(batch.shape[0]):
    timeDensity = batch[i,0].numpy()
    fockNote = batch[i,1:]
    fockIdx = maxFockIdx(fockNote,from_logits)
    newNotes = []
    for j in range(fockIdx + 1):
      thisNote = fockNote[fockOffsets[fockIdx] + j * singleNoteSize : fockOffsets[fockIdx] + (j + 1) * singleNoteSize]
      pitchPart = thisNote[:12]
      octavePart = thisNote[12:]

      # Here's where we forget anything with less than maximal probability
      bestPitch = np.argmax(pitchPart)
      bestOctave = np.argmax(octavePart)

      chosenMidiNote = 12 * bestOctave + bestPitch
      if chosenMidiNote > 0:
        newNotes.append(chosenMidiNote)

    for oldNote in heldNotes[:]:
      if oldNote not in newNotes:
        heldNotes.remove(oldNote)
        track.append(Message('note_off', note=min(oldNote,127), velocity=127, time=0))
      else:
        newNotes.remove(oldNote)

    for newNote in newNotes:
      #print("Outputing note",newNote)
      track.append(Message('note_on', note=min(newNote,127), velocity=64, time=0))
      heldNotes.append(newNote)

    track[-1].time = 16 * max(int(timeDensity),1)

  mid.save(title + '.mid')

# Helper to extract the most likely fock index (= note count - 1) from a
# fock probability vector that might be using logits
def maxFockIdx(fockNote, from_logits = True):
  probs = []
  for i in range(fockSize):
    if from_logits:
      probs.append(np.sum(tf.nn.softmax(fockNote[fockOffsets[i] : fockOffsets[i + 1]]).numpy()))
    else:
      probs.append(np.sum(fockNote[fockOffsets[i] : fockOffsets[i + 1]]))
  return np.argmax(probs)

# Display a sequence of NoteSpec's by printing it out, graphing it in a scatter plot, and writing a file
def displayBatch(batch,from_logits=True,title="batch"):
  print("Time densities:")
  print(batch[:,0].numpy())
  if from_logits:
    probs = tf.nn.softmax(batch[:,1:]).numpy()
  else:
    # Add epsilon to not break the plot axis
    probs = batch[:,1:]
  plt.scatter(range(len(probs[-1])),probs[-1],label=title)
  plt.yscale('log')
  writeBatch(batch,from_logits)

# Display some information about the model with
# its current parameters. Shows an example from the dataset
# and how well it is predicted by the model
def displayAbout(model,title="Model"):
  for exampleInput, target in validationSet.take(1):
    predict = model(exampleInput)

    #print("Input:")
    #displayBatch(exampleInput[0],from_logits=False,title="Input")
    #print()

    print("Target:")
    displayBatch(target[0],from_logits=False,title="Target")
    print()

    print("Predictions:")
    displayBatch(predict[0],from_logits=True,title="Predictions")
    print()

    plt.title(title)
    plt.legend()
    plt.show(block=True)

    # Plot the distribution of time density predictions over the whole batch vs the target
    # They should match for good networks, and if it just learns the time density mean, that
    # tells you it's bad
    plt.title("Time density distribution comparison for " + title)
    plt.hist([target[0][:,0],predict[0][:,0]],label=["Target time density distribution","Predicted time density distribution"])
    #plt.vlines([np.mean(input_example_batch[0][:,0])],label="Mean actual time density")
    plt.legend()
    plt.show(block=True)

# The big loss function
# Everything in here operates on symbolic, differentiable tensors
# So the code is quite constrained
# It also takes in an entire batch at once, hence all the [:,:,x] stuff
def lossfn(actual, pred):

  # First, compute the squared error loss for the time density
  predTimeDensity = pred[:,:,0]
  actualTimeDensity = actual[:,:,0]
  timeLoss = (predTimeDensity - actualTimeDensity) * (predTimeDensity - actualTimeDensity)

  #timeShapeLoss = tf.keras.losses.kullback_leibler_divergence(actualTimeDensity,predTimeDensity)

  # Next, compute the categorical entropy stuff
  # Firstly, for the fock index, and secondly for the pitch class and octave therein
  fockProbsAct = []
  fockProbsPre = []
  cce = []
  for i in range(fockSize):
    startIdx = 1 + fockOffsets[i]
    endIdx = 1 + fockOffsets[i + 1]

    logSoftPred = -tf.keras.backend.log(tf.nn.softmax(pred[:,:, startIdx : endIdx]))

    fockProbsAct.append(tf.keras.backend.sum(actual[:,:,startIdx : endIdx],axis=2))
    fockProbsPre.append(tf.keras.backend.sum(logSoftPred,axis=2))

    chunkOnNotesAct = tf.stack(tf.split(actual[:,:,startIdx : endIdx],singleNoteSize,axis=2),axis=0)
    chunkOnNotesPred = tf.stack(tf.split(logSoftPred,singleNoteSize,axis=2),axis=0)

    symmedAct = tf.keras.backend.mean(chunkOnNotesAct,axis=0)
    symmedPred = tf.keras.backend.mean(chunkOnNotesPred,axis=0)

    # Hacky way to make it compute the p \cdot log q stuff

    # Make a stack on axis=0 of [actual,log(softmax(pred))], then tf.prod along axis=0
    stacked = tf.stack([symmedAct, symmedPred],axis=0)
    cce.append(tf.keras.backend.sum(tf.keras.backend.prod(stacked,axis = 0),axis=2))

  fockProbsActTensor = tf.keras.backend.stack(fockProbsAct,axis=2)
  fockProbsPreTensor = tf.keras.backend.stack(fockProbsPre,axis=2)
  cceTensor = tf.keras.backend.stack(cce,axis=2)
  actualFockIdx = tf.math.argmax(fockProbsActTensor,axis=2)

  # This is the entropy from correctly categorizing the number of notes to play simultaneously
  fockIndexEntropy = -tf.math.log(tf.gather(fockProbsPreTensor,actualFockIdx,batch_dims=2))

  cceEntropy = tf.gather(cceTensor,actualFockIdx,batch_dims=2)

  alpha = 0.2
  return alpha * timeLoss + fockIndexEntropy + cceEntropy

model.compile(optimizer='adam', loss=lossfn, metrics=['accuracy'])


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# If the checkpoint directory exists, try to load weights from it
# To not use the old checkpoint, just delete the directory
if os.path.isdir(checkpoint_dir):
  model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

def sampleForwardModel(model,n=50):
  for inp, tar in validationSet.take(1):
    x = inp
    for i in range(50):
      x = model(x)
    displayBatch(x)

displayAbout(model,title="Existing model")
sel = input("Go?")

if len(sel) > 0 and sel[0] == "g":
  sampleForwardModel(model)

if sel != "y":
  quit()

model.fit(dataset, epochs=5, callbacks=[checkpoint_callback])

print(model.evaluate(validationSet))
displayAbout(model,title="Newly trained model")

