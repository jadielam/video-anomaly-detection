import os
import sys
import json

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    #1. Read video from video stream

    #2. Training phase
    #2.1 Pass video frame to network and get loss and logits
    #2.2 Keep loss and accuracy of last n frames
    #2.3 As soon as loss and accuracy reach a certain plateau or a number of predefined steps is reached, 
    #jump to evaluation phase

    #3. Evaluation phase:
    #3.1 Pass video frame to network and accuracy
    #3.2 As soon as logits value for a certain past k-window drop from a certain value
    # mark the event as an anomaly.

if __name__ == "__main__":
    main()