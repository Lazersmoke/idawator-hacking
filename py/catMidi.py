#!/usr/bin/python

from mido import MidiFile
import sys

lastBatchFile = MidiFile(sys.argv[1])
for i, track in enumerate(lastBatchFile.tracks):
  for msg in track:
    print(msg)
