from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker():
    def __init__(self,maxDist):
        """Initiate the centroid tracker class
        
            Args: 
            maxDist: the maximum distance between PSFs to link between consecutive frames"""
        self.maxDist = maxDist
        self.tracks = OrderedDict()
        self.total_tracks = OrderedDict()
        self.framenum = 0
        self.pointer = []
        self.centroid = []
    def track(self,new_centroid,framenum):
        """Perform tracking of PSFs according to minimum distance between localizations in consecutive frames
            
            Args:
                new_centroid: the newly localized PSFs in the currect frame
                framenum: the frame number
            Returns:
                Updates the pointer (tracking between frames) of the centroid tracker class and saves the localizations of the current frame"""
        self.framenum = framenum
        if self.framenum == 0:
            self.pointer = np.zeros((len(new_centroid)))
            self.centroid = new_centroid
            for i in range(0,len(new_centroid)):
                self.pointer[i] = i
                self.tracks[i] = np.array((new_centroid[i][0],new_centroid[i][1],self.framenum))
        else:
            D = dist.cdist(np.array(self.centroid), new_centroid)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            new_pointer = np.zeros((len(new_centroid))) -1
            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue
                if D[row,col]>self.maxDist:
                    usedRows.add(row)
                    usedCols.add(col)
                    continue
                else:
                    next_track = new_centroid[col]
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
                self.tracks[next+n] = np.append(new_centroid[i],self.framenum)
                new_pointer[i] = next+n
                n+=1
            self.pointer = new_pointer
            self.centroid = new_centroid







