'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

'''

import numpy as np
import cv2

from detect import detect_text

net = cv2.dnn.readNet("frozen_east_text_detection.pb")

lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    

feature_params = dict( maxCorners = 500, 
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 50
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

        cap = cv2.VideoCapture(video_src)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        videofile = "output.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(videofile, fourcc, fps, size)

    def run(self):
        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            if self.frame_idx % self.detect_interval == 0:
                newW = 320
                newH = 320
                (H, W) = frame.shape[:2]
                rW = W / float(newW)
                rH = H / float(newH)
                
                print("Text detect {}".format(self.frame_idx))
                text_regions = detect_text(frame, net, rW, rH, newW, newH)

                for text_region in text_regions:
                    x, y, w, h = text_region
                    
                    mask = np.zeros_like(frame_gray)
                    mask[y:y+h, x:x+w] = 255
                    
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                    
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            self.out.write(vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

def main():

    video_src = "input/news10_640x360.y4m"

    App(video_src).run()
    cv2.destroyAllWindows() 			

if __name__ == '__main__':
    main()
