from processing import Processing
from CentroidTracker import CentroidTracker
import numpy as np
from scipy import signal
import cv2
from skimage.feature import peak_local_max
from collections import OrderedDict
import pickle
from matplotlib import pyplot as plt

ULMinfo = dict(path = 'C:\\Users\\admin\\Desktop\\Data\\22.11.10 - 45 degree 300 micron phantom\\MBs_3\\MBs_',
supermats = range(1,3),
FR = 200,
num_bubbles = 50,
fwhmx = 0.25,
fwhmy = 0.075,
fovx = [-6.912, 6.912],
fovy = [15, 24])

def localization(path,supermats,FR,num_bubbles,fwhmx, fwhmy, fovx, fovy, SVD=True,display_on=True,tracking=True):

    """Perform ULM localization
    
        Args:
            path: the folder path to the data
            supermats: number of blocks of image data recorded
            FR: frame rate of recording
            SVD: boolean, svd filter the data or not
            display_on: boolean, whether to display localizations as a video while processing
            tracking: boolean, whether to perform tracking of PSFs by minimum distance
        Returns:
            ct.total_tracks: dictionary of all tracks in the form of [x,y,framenum]
            saves all tracks using pickle so image construction can be preformed without repeating localization"""

    processing = Processing()
    ct = CentroidTracker(3)
    for s in supermats:
        data_unfilt = processing.loaddata(path,s)
        if SVD:
            data_unfilt = processing.SVD_filt(data_unfilt,2)
        sos = signal.butter(4, [2, int((FR-10)/2)] , fs = FR, btype='band',output = 'sos')
        data = signal.sosfilt(sos, data_unfilt)
        data = data[:,:,10:]
        num_frames = data.shape[2]

        for i in range(0,num_frames):
            im = data[:,:,i]
            im = processing.log_scale(im, db=1)
            im_filt = cv2.fastNlMeansDenoising(im, None, 7, 5, 11)
            coordinates = peak_local_max(image = im_filt,min_distance=3,num_peaks = num_bubbles) 
            disp_im = cv2.merge([im_filt,im_filt,im_filt])
            coords_localized = processing.weighted_av(im_filt,coordinates,fwhmy,fwhmx,fovy,fovx)
            disp_im[np.round(coords_localized[:,0]).astype(int),np.round(coords_localized[:,1]).astype(int),:] = [0,0,255]
            if display_on:
                cv2.imshow('Frame Display',disp_im)
                cv2.waitKey(20)
            if tracking:
                ct.track(coords_localized,i)
        if tracking:
            ct.total_tracks[s] = {k: v for k, v in ct.tracks.items() if len(v) > 5}
    pickle.dump([ct.total_tracks,im.shape], open(path+"_all_tracks.p", "wb" ))
    return ct.total_tracks, im.shape

def tracks2output(path,supermats,FR,num_bubbles,fwhmx,fwhmy, fovx, fovy,tracks_dict,min_track_length,im_shape,scale,interpolate=False):

    """Form a super-resolved image and velocity map using the tracks dictionary
    
    Args: 
        tracks_dict: dictionary containing all tracks calculated for data
        min_track_length: minimal track length to use. Bubbles in tracks under this length are not used for reconstruction
        im_shape: shape of original data (in pixels)
        FR: frame rate of acquisition
        scale: how large to upscale the image (1- no upscaling)
        fovx: data size along x axis in mm. Example: [-6,6]
        fovy: data size along depth axis in mm. Example: [5,16]
        
    Returns:
        superres: super-resolved ULM image
        velocity_map: map of calculated velocity"""

    ppmx = ((fovx[1]-fovx[0]))/im_shape[1]
    ppmy = ((fovy[1]-fovy[0]))/im_shape[0]
    output = np.zeros(im_shape*scale)
    vel_y = np.zeros(im_shape*scale)
    vel_x = np.zeros(im_shape*scale)
    for k,v in tracks_dict.items():
        for track_num, track in v.items():
            if len(track) > min_track_length:
                roundx = np.round(track[:,0]).astype(int)
                roundy = np.round(track[:,1]).astype(int)
                output[roundx,roundy] += 1
                av_vel_x_pix = np.mean(np.diff(roundx))
                av_vel_y_pix = np.mean(np.diff(roundy))
                vel_y[roundx,roundy] += av_vel_y_pix*FR*ppmx
                vel_x[roundx,roundy] += av_vel_x_pix*FR*ppmy
        vel_y[output>0] /= output[output>0]
        vel_x[output>0] /= output[output>0]
        velocity = np.sqrt(vel_x**2+vel_y**2)
    return output,velocity

tracks_dict,im_shape = localization(**ULMinfo,SVD = True,display_on = True,tracking = True)
loaded_objects = pickle.load(open(ULMinfo['path']+"_all_tracks.p", "rb" ))
superres, velocity = tracks2output(**ULMinfo, tracks_dict = loaded_objects[0], min_track_length = 5, im_shape = loaded_objects[1], scale = 1, interpolate = False)
plt.figure(1)
plt.imshow(superres,cmap = 'hot')
plt.figure(2)
plt.imshow(velocity,cmap = plt.cm.nipy_spectral)
clb = plt.colorbar()
clb.ax.set_title('mm/sec')
plt.show()