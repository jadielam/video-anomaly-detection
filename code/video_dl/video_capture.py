import cv2
import config
from datetime import datetime

def video_capture_proc(frames_queue, conf):
    camera_url = conf['video_feed']['url']
    number_of_frames = conf['number_of_frames']
    cap = cv2.VideoCapture(camera_url)

    fid = -1
    while cap.isOpened():
        fid += 1
        try:
            ret, frame = cap.read()
        except:
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_url)
                print("{},ERROR: Problem reading video feed.\n".format(datetime.now()))
            continue
        
        if fid > number_of_frames:
            break
        
        if not ret:
            if cap.isOpened():
                cap.release()
            cap = cv2.VideoCapture(camera_url)
            print("{},ERROR: Problem reading video feed.\n".format(datetime.now()))
        else:

            #Here we drop frames if the queue is full.
            try:
                frames_queue.put_nowait((fid, frame))
            except:
                pass
    
    cap.release()
    frames_queue.put(config.FINISH_SIGNAL)
