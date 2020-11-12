# Irrelevant long comment that has nothing to do with the contents of this file:

# Want to manipulate your sample as midi. Functor:
# Midi A ------> Midi A'
#   ^              |
#   |              |
#   |              V
# Sample A ----> Sample A'


# What's in a note? It's a function <Sample Data> -> <Subset of Sample Data>
# "Go to times [0.5,0.7], filter for these frequencies, take whatever you find there, that's the note"
# Can be overlapping (trivially)

# Composable! One note is "times [0.5,0.7]", another is "filter for [lo,hi] frequencies", another is "envelope"

# Can have "constant note" which just samples from a midi instrument or rack controller or what have you

# Decompose the ADSR into:
# - Attack note, including configurable amount of decay
# - Entire note, with sustain
# - Sustain "minus" attack
# - etc



import magenta
import note_seq
import tensorflow as tf

print("Versions")
print(magenta.__version__)
print(tf.__version__)

from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

import os

# Initialize the model.
print("Initializing Music VAE...")
music_vae = TrainedModel(
      configs.CONFIG_MAP['cat-mel_2bar_big'], 
      batch_size=4, 
      checkpoint_dir_or_path=os.path.join(os.getcwd(), 'cat-mel_2bar_big.ckpt'))

dimZ = 512

def modelDecode(zs):
  return music_vae.decode(zs,length=80,temperature=0.001)

# Version of model decode that runs in O(1) by just returning the same note sequence every time
fakeNS = modelDecode(tf.random.normal([1,dimZ]))
def fakeDecode(zs):
  return fakeNS * tf.shape(zs)[0].numpy()


def differentiate(baseZ,func,h=0.3):
  fzero = func(modelDecode(tf.expand_dims(baseZ,0))[0])
  units = h * tf.eye(dimZ)
  unitDisplacedZs = tf.add(baseZ,units)
  decodedUnitDisplaced = modelDecode(unitDisplacedZs)
  return [(func(x) - fzero)/h for x in decodedUnitDisplaced]


# Fake test function to differentiate (should have gradient zero)
def constOne(x):
  return 1

def noteCount(ns):
  print(ns)
  noteCount = 0
  for note in ns.notes:
    noteCount += 1
  print(noteCount)
  return noteCount

print("going")
samps = music_vae.sample(n=1, length=80, temperature=1.0)
pointInSpace = tf.squeeze(music_vae.encode(samps)[0])
tf.print(pointInSpace)
print(tf.shape(pointInSpace))
print(differentiate(pointInSpace,noteCount))

