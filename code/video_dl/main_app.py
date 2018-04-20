import os
import sys
import json
from multiprocessing import Process, Queue

from video_capture import video_capture_proc
from anomaly_detection import anomaly_detection_proc

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    #1. Read video from video stream
    frames_queue = Queue(1)
    proc_video_capture = Process(target = video_capture_proc, 
                                args = (frames_queue, conf))

    proc_anomaly_detection = Process(target = anomaly_detection_proc, 
                                args = (frames_queue, conf))

    proc_video_capture.start()
    proc_anomaly_detection.start()
    proc_video_capture.join()
    proc_anomaly_detection.join()

if __name__ == "__main__":
    main()