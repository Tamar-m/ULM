from collections import OrderedDict
import numpy as np
import cv2
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt 
import math



class OpticFlow():

    def __init__(self,max_dist = 1):
        self.tracks = OrderedDict()
        self.total_tracks = OrderedDict()
        self.framenum = 0
        self.old_frame = []
        self.init_points = []
        self.pointer = []
        self.max_dist = max_dist
        self.lk_params = dict( winSize  = (5,5), 
                  maxLevel = 1, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
        arrow = np.zeros((5,5), dtype=np.uint8)
        for i in range(5):
            for j in range(5):
                if i-j ==3 or i+j == 3:
                    arrow[i,j] = 1
        self.arrow = arrow
    def track(self,new_frame,new_points,framenum): 
        self.framenum = framenum
        if self.framenum == 0:
            self.pointer = np.zeros((len(new_points)))
            self.init_points = new_points
            mask = cv2.adaptiveThreshold(new_frame,255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,17,-8)  
            new_frame = np.zeros_like(new_frame)
            new_frame[np.floor(new_points[:,0]).astype(int),new_points[:,1].astype(int)] = 255 
            new_frame[mask==0] = 0
            new_frame = cv2.dilate(new_frame, self.arrow)
            self.old_frame = new_frame
            for i in range(0,len(new_points)):
                self.pointer[i] = i
                self.tracks[i] = np.expand_dims(np.array((new_points[i][0],new_points[i][1],self.framenum)),0)
                
        else:
            mask = cv2.adaptiveThreshold(new_frame,255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,17,-10)  
            new_frame = np.zeros_like(new_frame)
            new_frame[np.floor(new_points[:,0]).astype(int),new_points[:,1].astype(int)] = 255 
            new_frame[mask==0] = 0
            new_frame = cv2.dilate(new_frame, self.arrow)

            # plt.figure()
            # plt.imshow(self.old_frame)
            # plt.show()
            of_points, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame.astype(np.uint8),
                                                           new_frame.astype(np.uint8),
                                                             self.init_points.reshape(-1, 1, 2).astype(np.float32),
                                                               None, **self.lk_params)
            
            of_points[st == 0] = np.inf
            of_points = of_points.reshape(-1, 2)
            of_points = np.flip(of_points)
            

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
                    # test1 = self.tracks[pnt][-1][0:2]
                    # test2 = next_track[0:2]
                    # test3 = math.dist(test1,test2)
                    if math.dist(self.tracks[pnt][-1][0:2],next_track[0:2]) <= 3:
                        self.tracks.update({pnt:np.vstack((self.tracks[pnt],(next_track)))})
                    usedRows.add(row)
                    usedCols.add(col)
            ind = np.argwhere(new_pointer == -1)
            next = int(max(self.tracks.keys()))
            n = int(1)
            for i in ind:
                self.tracks[next+n] = np.expand_dims(np.append(new_points[i],self.framenum),0)
                new_pointer[i] = next+n
                n+=1
            self.pointer = new_pointer
            self.init_points = np.flip(new_points)
            self.old_frame = new_frame





