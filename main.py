from processing import Processing
from CentroidTracker import CentroidTracker
from OpticFlow import OpticFlow
import numpy as np
import scipy as sc
import cv2
import pickle
from matplotlib import pyplot as plt
import scipy
from scipy.io import savemat
from scipy.signal import savgol_filter

def localization(path,supermats,FR,num_bubbles,fovx, fovy, SVD=True,display_on=True,tracking='CT', loc_method = 'brightest_points'):

    """Perform ULM localization
    
        Args:
            path: the folder path to the data (including start of numbered super-frame name- example: MBs_ for super frames MBs_1,MBs_2,...)
            supermats: number of blocks of image data recorded
            FR: frame rate of recording
            num_bubbles: estimated number of bubbles per frame, relevant only if loc_method == 'brightest points' (may work better on in vivo data)
            fovx: data size along x axis in mm. Example: [-6,6]
            fovy: data size along depth axis in mm. Example: [5,16]
            SVD: boolean, svd filter the data or not
            display_on: boolean, whether to display localizations as a video while processing
            tracking: 'CT' for centroid tracker, 'optic_flow' for optic flow estimation
            loc_method: 'brightest_points' for brightest points in image or 'adaptive_thresh' for adaptive threshold per frame
        Returns:
            ct.total_tracks: dictionary of all tracks in the form of [x,y,framenum], dictionary key is arbitrary track number
            saves all tracks using pickle so image construction can be preformed without repeating localization"""

    processing = Processing()
    ct = CentroidTracker(4)
    OptFlow = OpticFlow(4)
    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("./models/FSRCNN_x2.pb")
    sr.setModel("fsrcnn", 2)
    for s in supermats:
        # s = '{0:03}'.format(s) # for loading data in superframes of more than 100 and numbering is 001, 002, 003 ... etc
        data_unfilt = processing.loaddata(path,s)
        if SVD:
            data_unfilt = processing.SVD_filt(data_unfilt,4)
        sos = sc.signal.butter(2, [10, int((FR-20)/2)] , fs = FR, btype='band',output = 'sos') 
        data = sc.signal.sosfilt(sos, data_unfilt)
        data = data[:,:,10:]
        num_frames = data.shape[2]

        for i in range(0,num_frames):
            im = data[:,:,i]
            im = processing.log_scale(im, db=1)
            im_upsample = scipy.ndimage.zoom(im, 2, order=1)

            if loc_method == 'brightest_points':  
                im_filt = cv2.fastNlMeansDenoising(im_upsample, None, 7, 5, 11)
                coords_localized = processing.brightest_points(image = im_filt, num_bubbles = num_bubbles)
            if loc_method == 'adaptive_thresh':
                im_filt = sr.upsample(im)
                coords_localized = processing.adaptive_thresh(image = im_filt)

            disp_im = cv2.merge([im_upsample,im_upsample,im_upsample])
            disp_im[np.round(coords_localized[:,0]).astype(int),np.round(coords_localized[:,1]).astype(int),:] = [0,0,255]
            if display_on:
                cv2.imshow('Frame Display',disp_im)
                cv2.waitKey(10)
            if tracking == 'CT':
                ct.track(coords_localized,i)
            elif tracking == 'optic_flow':
                OptFlow.track(im_filt,coords_localized,i)
        if tracking == 'CT':
            ct.total_tracks[s] = {k: v for k, v in ct.tracks.items()} 
            total_tracks = ct.total_tracks
        elif tracking == 'optic_flow':
            OptFlow.total_tracks[s] = {k: v for k, v in OptFlow.tracks.items()} 
            total_tracks = OptFlow.total_tracks

    pickle.dump([total_tracks,im.shape,num_frames], open(path+"_all_tracks.p", "wb" ))
    return total_tracks, im_upsample.shape

def tracks2output(path,supermats,FR,num_bubbles, fovx, fovy,tracks_dict,min_track_length,im_shape,num_frames, scale,interpolate=False):

    """Form a super-resolved image and velocity map using the tracks dictionary
    
    Args: 
        path: the folder path to the data (including start of numbered super-frame name- example: MBs_ for super frames MBs_1,MBs_2,...)
        supermats: number of blocks of image data recorded
        FR: frame rate of acquisition
        num_bubbles: estimated number of bubbles per frame, relevant only if loc_method == 'brightest points'
        fovx: data size along x axis in mm. Example: [-6,6]
        fovy: data size along depth axis in mm. Example: [5,16]
        tracks_dict: dictionary containing all tracks calculated for data
        min_track_length: minimal track length to use. Tracks under this length are not used for reconstruction of velocity maps
        im_shape: shape of original data (in pixels)
        scale: how large to upscale the image- even numbers (2,4,6...)
        num_frames: number of frames in each supermat
        scale: scale to upsample the final superresolved image (1: no upsampling)
        interpolate: whether or not to interpolate the final tracks in the super-resolved image

        
    Returns:
        superres: super-resolved ULM image
        velocity: map of calculated velocity
        DensityIm_time: accumulated localizations every 100 frames, for processing data over time"""
    
    scale /= 2
    scale = int(scale)
    im_shape = np.array(im_shape)*scale*2
    ppmx = ((fovx[1]-fovx[0]))/im_shape[1]
    ppmy = ((fovy[1]-fovy[0]))/im_shape[0]
    output = np.zeros(im_shape)
    vel_counts = np.zeros(im_shape)
    vel_y = np.zeros(im_shape)
    vel_x = np.zeros(im_shape)
    DensityIm_time = np.zeros((im_shape[0],im_shape[1],(num_frames+30)//100*supermats[-1]))
    for k,v in tracks_dict.items():
        for track_num, track in v.items():
            if track.shape == (3,):
                track = np.expand_dims(track, axis=0)
            if len(track) > min_track_length:
                interp_x = Processing.interp(track[:,0]*scale)
                interp_y = Processing.interp(track[:,1]*scale)
                roundx = np.round(interp_x).astype(int)
                roundy = np.round(interp_y).astype(int)
                pairs = np.stack((roundx,roundy),axis = -1)
                arr, uniq_cnt = np.unique(pairs, axis=0, return_counts=True)
                roundx = arr[:,0]
                roundy = arr[:,1] 

                roundx_raw = np.round(track[:,0]*scale).astype(int)
                roundy_raw = np.round(track[:,1]*scale).astype(int)
                DensityIm_time[roundx_raw,roundy_raw,((track[:,2]//100)+(k-1)*num_frames//100).astype(int)] += 1
                
                l = len(track[:,0])
                av_vel_x_pix =  np.mean(np.diff(savgol_filter(track[:,0],min((l- np.mod(l-1,2)),5),2))) # scale*(track[-1,0] - track[0,0])/len(track[:,0]) # 
                av_vel_y_pix =   np.mean(np.diff(savgol_filter(track[:,1],min((l- np.mod(l-1,2)),5),2))) # scale*(track[-1,1] - track[0,1])/len(track[:,1]) # 
                # if (av_vel_y_pix > 0): # and (av_vel_x_pix > 0): # used this for some difficut data to improve velocity map because we know the flow direction
                vel_y[roundx, roundy] += av_vel_x_pix*FR*ppmx # the velocity tracks are always interpolated
                vel_x[roundx, roundy] += av_vel_y_pix*FR*ppmy
                vel_counts[roundx, roundy] += 1
            else: # here we also add the single bubbles to final superresolution image (even if they could not be tracked)
                roundx_raw = np.round(track[:,0]*scale).astype(int)
                roundy_raw = np.round(track[:,1]*scale).astype(int)
            # if interpolate is true, add interpolated tracks to superresolution image
            if interpolate & len(track) > 1:
                output[roundx,roundy] += 1
            else:
                output[roundx_raw,roundy_raw] += 1
                DensityIm_time[roundx_raw,roundy_raw,((track[:,2]//100)+(k-1)*num_frames//100).astype(int)] += 1

    vel_y[vel_counts>0] /= vel_counts[vel_counts>0]
    vel_x[vel_counts>0] /= vel_counts[vel_counts>0] 
    vel_y2 = cv2.GaussianBlur(vel_y, (3,3), 0)
    vel_x2 = cv2.GaussianBlur(vel_x, (3,3), 0)
    velocity = np.hypot(vel_x2,vel_y2)
    for i in range(2,np.shape(DensityIm_time)[2]):
        DensityIm_time[:,:,i] += DensityIm_time[:,:,i-1]
    return output,vel_x,vel_y,velocity,DensityIm_time


ULMinfo = dict(path = 'C:\\Users\\admin\\Desktop\\Data\\22.12.28 - different width phantoms\\300_100\\flow rate 0.032 FR 80 V 20 1\\MBs_',
supermats = range(1,6), # this is the number of data blocks you have (for example, 4 blocks of 1500 frames)
FR = 100, 
num_bubbles = 40, 
fovx = [-6.912, 6.912],  
fovy = [15, 22]) 

tracks_dict, im_shape = localization(**ULMinfo,SVD = True, display_on = True,tracking = 'optic_flow', loc_method = 'adaptive_thresh')


loaded_objects = pickle.load(open(ULMinfo['path']+"_all_tracks.p", "rb" ))
superres, vel_x, vel_y, velocity, DensityIm_time = tracks2output(**ULMinfo, tracks_dict = loaded_objects[0], min_track_length = 3,
                                                                  im_shape = loaded_objects[1], num_frames = loaded_objects[2], scale = 2,
                                                                    interpolate = True)

experiment_results = {"superres":superres, "velocity":velocity, "DensityIm_time":DensityIm_time,"im_shape":loaded_objects[1]}
savemat(ULMinfo['path'] + "experiment_results.mat", experiment_results) 
savemat(ULMinfo['path'] + "experiment_info.mat", ULMinfo)

plt.figure(1)
plt.imshow(superres**(1/2),cmap = 'hot')
plt.savefig(ULMinfo['path'] + "superresolution.png")
plt.axis('off')

plt.figure(2)
plt.imshow(velocity,cmap = plt.cm.nipy_spectral)
plt.savefig(ULMinfo['path'] + "velocitymap.png")
plt.axis('off')
clb = plt.colorbar()
clb.ax.set_title('mm/sec')


plt.show()
