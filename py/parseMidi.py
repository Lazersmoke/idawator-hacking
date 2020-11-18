import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import wavfile
from scipy.signal import hilbert
from scipy.special import binom
from mido import MidiFile
import itertools

import nonlin

fockSize = 10
# Pitch class, octave
sizeOfNoteSpec = 12 + 1
sizeOfFockNoteSpec = 0
fockOffsets = []
for k in range(fockSize + 1):
  if k < fockSize:
    fockOffsets.append(sizeOfFockNoteSpec + sizeOfNoteSpec * k)
  sizeOfFockNoteSpec += sizeOfNoteSpec * k

print("Fock size total:",sizeOfFockNoteSpec)
print("Fock offsets:",fockOffsets)
# Include time density!
sizeEpoch = 1 + sizeOfFockNoteSpec

ohbMatrix = np.eye(12)

# Build a NoteSpec out of the current midi situation during this particular epoch
def mkNoteSpec(heldNotes,decayingNotes,timeDensity):
  allNotes = heldNotes + decayingNotes
  noteCount = len(allNotes)
  if noteCount > fockSize:
    print("!!! Warning, fock size of {} exceeded by {} simultaneous notes !!!".format(fockSize,noteCount))
    allNotes = allNotes[:fockSize]
    noteCount = fockSize
  fOff = fockOffsets[noteCount - 1]
  fockVec = np.zeros(sizeOfFockNoteSpec)

  contribs = []
  for k in range(noteCount):
    (octave,pc) = midiNoteToRepr(allNotes[k])
    pcVec = np.zeros(12)
    pcVec[pc] = 1
    nOff = fOff + k * sizeOfNoteSpec
    fockVec[nOff : nOff + sizeOfNoteSpec] = np.append(np.matmul(ohbMatrix,pcVec),octave)
  #print(pc)
  epoch = np.insert(fockVec,0,timeDensity)
  return epoch

def traceNoteSpec(ns):
  mess = "Time Density: {}, note probabilites:".format(ns[0])
  for k in range(fockSize):
    s = 1 + fockOffsets[k]
    fock = ns[s : s + (k + 1) * sizeOfNoteSpec]
    prob = np.linalg.norm(fock)
    if prob > 0:
      mess += "\n{:.2f} for {} notes (".format(prob,k + 1)
      for l in range(k + 1):
        # minus one to forget octave
        noteStart = l * sizeOfNoteSpec
        thisNote = fock[noteStart : noteStart + sizeOfNoteSpec - 1]
        thisOctave = fock[noteStart + sizeOfNoteSpec - 1]
        mess += "{}^{}, ".format(np.argwhere(thisNote).flatten(),thisOctave)
      mess = mess[:-2] + ")"
  return mess

# Midi should have octave in integers [0,10] (so eleven octaves)
# Returns (octave,pitchClass)
def midiNoteToRepr(midiNote):
  return divmod(midiNote,12)

mid = MidiFile('stayorgo.mid')

tracks = []
for i, track in enumerate(mid.tracks):
  print('Track {}: {}'.format(i, track.name))
  heldNotes = []
  toUnHold = []
  lastTime = 0
  noteSpecs = []
  for msg in track:
    if msg.time != 0:
      #print()
      #print("Held",heldNotes,"with these ones decaying:",toUnHold,"for time:",lastTime)
      #print()
      noteSpecs.append(mkNoteSpec(heldNotes,toUnHold,min(lastTime,300)))
      #print(traceNoteSpec(noteSpecs[-1]))
      for n in toUnHold:
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
  tracks.append(np.stack(noteSpecs,axis=0))
  print("Found {} Note Specs\n".format(len(noteSpecs)))


# Memoryless predictor network
def stepPredictLoss(predictor,track):
  print("Finding step loss...")
  totalLoss = 0
  lastNoteSpec = None
  for noteSpec in track:
    if lastNoteSpec is not None:
      predicted = nonlin.applyNonlinear(predictor,lastNoteSpec)
      totalLoss += np.linalg.norm(predicted - noteSpec)
    lastNoteSpec = noteSpec
  return totalLoss

def timePredictLoss(predictor,track):
  totalLoss = 0
  for k in range(track.shape[0] - predictorSideLength):
    i = k + predictorSideLength
    predicted = nonlin.applyNonlinear(predictor,track[k:i,0])[0]
    totalLoss += np.abs(predicted - track[i,0]) ** 2
  return np.log(totalLoss)

def plotTimePredictLoss(predictor,track):
  totalLoss = 0
  losses = []
  ks = range(track.shape[0] - predictorSideLength)
  for k in ks:
    i = k + predictorSideLength
    predicted = nonlin.applyNonlinear(predictor,track[k:i,0])[0]
    totalLoss += np.abs(predicted - track[i,0]) ** 2
    losses.append(totalLoss)
  return losses



predictorDepth = 2
predictorSideLength = 3
identityPredictor = nonlin.identityNonLin(sizeEpoch,predictorDepth)
identityTimePredictor = nonlin.identityNonLin(predictorSideLength,predictorDepth)
print("Computing identity time loss on Track 1...")
print("Total Loss",timePredictLoss(identityTimePredictor,tracks[1]))

def minCB(params):
  predictor = nonlin.deserializeNonLin(params,predictorSideLength,predictorDepth)
  print(predictor)
  for l in range(10):
    k = l + 200
    i = k + predictorSideLength
    predicted = nonlin.applyNonlinear(predictor,tracks[1][k:i,0])[0]
    print("Predicted: ",predicted)
    print("Actual: ",tracks[1][i,0])
    print("Loss: ",np.abs(predicted - tracks[1][i,0]) ** 2)
    print()
  print("Step Loss: ",timePredictLoss(predictor,tracks[1]))
  allLosses.append(plotTimePredictLoss(predictor,tracks[1]))

def toMinimize(params):
  predictor = nonlin.deserializeNonLin(params,predictorSideLength,predictorDepth)
  return timePredictLoss(predictor,tracks[1])

input("...")
allLosses = []
minResult = minimize(toMinimize,nonlin.serializeNonLin(identityTimePredictor),callback=minCB,options={'disp':True,'maxiter':10})
print(minResult)

for i in range(len(allLosses)):
  losses = allLosses[i]
  plt.plot(range(len(losses)),losses, color = str(1-(i/len(allLosses))))

plt.title('Cumulative loss during track, progressive iterations')
plt.xlabel('Note in Track')
plt.ylabel('Cumulative Loss')
plt.show(block=True)

plt.hist(np.diff(allLosses[-1]))
plt.title('Distribution of losses')
plt.show(block=True)
