from tracemalloc import start
from muselsl import stream, list_muses, view, record
from pylsl import StreamInlet, resolve_stream
import os


def start_stream():
  streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
  inlet = StreamInlet(streams[0])

  while True:
    sample, timestamp = inlet.pull_sample()
    print(sample)

if __name__ == "__main__":
  start_stream()