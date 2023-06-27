from collections import OrderedDict
import numpy as np
import cv2
from scipy.spatial import distance as dist




class OpticFlow():

    def __init__(self,max_dist = 1):
        self.tracks = OrderedDict()
        self.total_tracks = OrderedDict()
        self.framenum = 0
        self.old_frame = []
        self.init_points = []
        self.pointer = []
        self.max_dist = max_dist
        self.lk_params = dict( winSize  = (7,7), 
                  maxLevel = 1, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    def track(self,new_frame,new_points,framenum):
        self.framenum = framenum


        if self.framenum == 0:
            self.pointer = np.zeros((len(new_points)))
            self.init_points = new_points
            self.old_frame = new_frame
            for i in range(0,len(new_points)):
                self.pointer[i] = i
                self.tracks[i] = np.array((new_points[i][0],new_points[i][1],self.framenum))
                
        else:
            test = self.init_points.reshape(-1, 1, 2).astype(np.float32)
            test2 = self.old_frame.astype(np.uint8)
            of_points, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame.astype(np.uint8), new_frame.astype(np.uint8), self.init_points.reshape(-1, 1, 2).astype(np.float32), None, **self.lk_params)
            of_points[st == 0] = np.inf
            of_points = of_points.reshape(-1, 2)

            D = dist.cdist(of_points, new_points)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            new_pointer = np.zeros((len(new_points))) -1
            for (row, col, ind) in zip(rows, cols, range(D.shape[0])):
                if row in usedRows or col in usedCols:
                    continue
                if D[row,col]>self.max_dist:
                    st[row] = 0
                    usedRows.add(row)
                    usedCols.add(col)
                    continue
                else:
                    next_track = new_points[col]
                    next_track = np.append(next_track,self.framenum)
                    pnt = self.pointer[row]
                    new_pointer[col] = pnt
                    self.tracks.update({pnt:np.vstack((self.tracks[pnt],(next_track)))})
                    usedRows.add(row)
                    usedCols.add(col)
            ind = np.argwhere(new_pointer == -1)
            next = int(max(self.tracks.keys()))
            n = int(1)
            for i in ind:
                self.tracks[next+n] = np.append(new_points[i],self.framenum)
                new_pointer[i] = next+n
                n+=1
            self.pointer = new_pointer
            self.init_points = new_points
            self.old_frame = new_frame





