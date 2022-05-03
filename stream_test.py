from pylsl import StreamInlet, resolve_stream, resolve_byprop
import os
import time
import numpy as np
import utils
import subprocess
import sys

BUFFER_LENGTH = 20
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
flag = True


def main():
    p = subprocess.Popen(["muselsl", "stream"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, encoding="utf-8", errors="replace")
    
    while True:
        output = p.stdout.readline()
        
        if output == "" and p.poll() is not None:
            break
        if output:
            print(output)
        
        sys.stdout.flush()
    print("done")
    

if __name__ == "__main__":
    main()