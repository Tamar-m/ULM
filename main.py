from processing import Processing
from CentroidTracker import CentroidTracker
from OpticFlow import OpticFlow
import numpy as np
import scipy as sc
import cv2
from skimage.feature import peak_local_max
from collections import OrderedDict
import pickle
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter

def localization(path,supermats,FR,num_bubbles,fwhmx, fwhmy, fovx, fovy, SVD=True,display_on=True,tracking='CT'):

    """Perform ULM localization
    
        Args:
            path: the folder path to the data (including start of numbered super-frame name- example: MBs_ for super frames MBs_1,MBs_2,...)
            supermats: number of blocks of image data recorded
            FR: frame rate of recording
            SVD: boolean, svd filter the data or not
            display_on: boolean, whether to display localizations as a video while processing
            tracking: 'CT' for centroid tracker, 'optic_flow' for optic flow
        Returns:
            ct.total_tracks: dictionary of all tracks in the form of [x,y,framenum]
            saves all tracks using pickle so image construction can be preformed without repeating localization"""

    processing = Processing()
    ct = CentroidTracker(3)
    OptFlow = OpticFlow(1)
    
    # sr = cv2.dnn_superres.DnnSuperResImpl_create()
    # sr.readModel("./models/FSRCNN_x2.pb")
    # sr.setModel("fsrcnn", 2)
    for s in supermats:
        # s = '{0:03}'.format(s)
        data_unfilt = processing.loaddata(path,s)
        if SVD:
            data_unfilt = processing.SVD_filt(data_unfilt,5)
        sos = sc.signal.butter(2, [10, int((FR-20)/2)] , fs = FR, btype='band',output = 'sos') #for our data int((FR-20)/2)
        data = sc.signal.sosfilt(sos, data_unfilt)
        data = data[:,:,10:]
        num_frames = data.shape[2]

        for i in range(0,num_frames):
            im = data[:,:,i]
            im = processing.log_scale(im, db=1)
            # im = sr.upsample(im)
            # im_filt = cv2.fastNlMeansDenoising(im, None, 7, 5, 11)
            im_filt = im
            coordinates = peak_local_max(image = im_filt,min_distance=6,num_peaks = num_bubbles) 
            disp_im = cv2.merge([im_filt,im_filt,im_filt])
            coords_localized = processing.weighted_av(im_filt,coordinates,fwhmy,fwhmx,fovy,fovx)
            disp_im[np.round(coords_localized[:,0]).astype(int),np.round(coords_localized[:,1]).astype(int),:] = [0,0,255]
            if display_on:
                cv2.imshow('Frame Display',disp_im)
                cv2.waitKey(10)
            if tracking == 'CT':
                ct.track(coords_localized,i)
            elif tracking == 'optic_flow':
                OptFlow.track(im_filt,coords_localized,i)
        if tracking == 'CT':
            ct.total_tracks[s] = {k: v for k, v in ct.tracks.items()} # if len(v) > 3
            total_tracks = ct.total_tracks
        elif tracking == 'optic_flow':
            OptFlow.total_tracks[s] = {k: v for k, v in OptFlow.tracks.items()} # if len(v) > 3
            total_tracks = OptFlow.total_tracks

    pickle.dump([total_tracks,im.shape,num_frames], open(path+"_all_tracks.p", "wb" ))
    return total_tracks, im.shape

def tracks2output(path,supermats,FR,num_bubbles,fwhmx,fwhmy, fovx, fovy,tracks_dict,min_track_length,im_shape,num_frames, scale,interpolate=False):

    """Form a super-resolved image and velocity map using the tracks dictionary
    
    Args: 
        tracks_dict: dictionary containing all tracks calculated for data
        min_track_length: minimal track length to use. Bubbles in tracks under this length are not used for reconstruction
        im_shape: shape of original data (in pixels)
        FR: frame rate of acquisition
        scale: how large to upscale the image (larger than 2)
        fovx: data size along x axis in mm. Example: [-6,6]
        fovy: data size along depth axis in mm. Example: [5,16]
        
    Returns:
        superres: super-resolved ULM image
        velocity: map of calculated velocity
        DensityIm_time: accumulated localizations every 100 frames, for processing data over time"""
    
    # scale /= 2
    scale = int(scale)
    im_shape = np.array(im_shape)*scale
    ppmx = ((fovx[1]-fovx[0]))/im_shape[1]
    ppmy = ((fovy[1]-fovy[0]))/im_shape[0]
    output = np.zeros(im_shape)
    vel_counts = np.zeros(im_shape)
    vel_y = np.zeros(im_shape)
    vel_x = np.zeros(im_shape)
    DensityIm_time = np.zeros((im_shape[0],im_shape[1],(num_frames+30)//100*supermats[-1]))
    for k,v in tracks_dict.items():
        test = tracks_dict.items()
        for track_num, track in v.items():
            if len(track) > min_track_length:
                if track.shape == (3,):
                    track = np.expand_dims(track, axis=0)
                interp_x = Processing.interp(track[:,0]*scale)
                interp_y = Processing.interp(track[:,1]*scale)
                # interp_x = savgol_filter(interp_x, 21, 7)
                # interp_y = savgol_filter(interp_y, 21, 7)
                roundx = np.round(interp_x).astype(int)
                roundy = np.round(interp_y).astype(int)
                pairs = np.stack((roundx,roundy),axis = -1)
                arr, uniq_cnt = np.unique(pairs, axis=0, return_counts=True)
                roundx = arr[:,0]
                roundy = arr[:,1] 

                roundx_raw = np.round(track[:,0]*scale).astype(int)
                roundy_raw = np.round(track[:,1]*scale).astype(int)
                # f, (ax1, ax2) = plt.subplots(1, 2)
                # ax1.set_title('Not interpolated')
                # ax1.scatter(roundx_raw, roundy_raw, c='blue', lw=0.1)
                # for i, txt in enumerate(track[:,2]):
                #     ax1.annotate(txt, (track[i,0], track[i,1]))
                # ax2.set_title('Interpolated')
                # ax2.scatter(roundx, roundy, c='blue', lw=0.1)
                # plt.show()
                DensityIm_time[roundx_raw,roundy_raw,((track[:,2]//100)+(k-1)*num_frames//100).astype(int)] += 1

                output[roundx,roundy] += 1
                l = len(track[:,0])
                av_vel_x_pix = (track[-1,0] - track[0,0])/len(track[:,0]) # np.mean(np.diff(savgol_filter(track[:,0],min((l- np.mod(l-1,2)),5),2))) #np.mean(np.diff(track[:,0])) 
                av_vel_y_pix = (track[-1,1] - track[0,1])/len(track[:,1]) # np.mean(np.diff(savgol_filter(track[:,1],min((l- np.mod(l-1,2)),5),2))) #np.mean(np.diff(track[:,1])) 
                if (av_vel_y_pix > 0): # and (av_vel_x_pix > 0):
                    vel_y[roundx, roundy] += av_vel_x_pix*FR*ppmx
                    vel_x[roundx, roundy] += av_vel_y_pix*FR*ppmy
                    vel_counts[roundx, roundy] += 1
            else: 
                roundx_raw = np.round(track[:,0]*scale).astype(int)
                roundy_raw = np.round(track[:,1]*scale).astype(int)
                output[roundx_raw,roundy_raw] += 1
        vel_y[vel_counts>0] /= vel_counts[vel_counts>0]
        vel_x[vel_counts>0] /= vel_counts[vel_counts>0]
        velocity = np.sqrt(vel_x**2+vel_y**2)
    for i in range(2,np.shape(DensityIm_time)[2]):
        DensityIm_time[:,:,i] += DensityIm_time[:,:,i-1]
    return output,velocity,DensityIm_time


ULMinfo = dict(path = 'C:\\Users\\admin\\Desktop\\Data\\23.07.16 - new phantoms\\microfluidic phantom\\attempt 1\\MBs_',
supermats = range(1,9), 
FR = 100,
num_bubbles = 20, 
fwhmx = .1, #for our transducer .025
fwhmy = .1, #for our transducer .075
fovx = [-6.912, 6.912], #[-75,75],  
fovy = [15, 22]) #[0,150]) 

# tracks_dict, im_shape = localization(**ULMinfo,SVD = True, display_on = True,tracking = 'optic_flow')


loaded_objects = pickle.load(open(ULMinfo['path']+"_all_tracks.p", "rb" ))
superres, velocity, DensityIm_time = tracks2output(**ULMinfo, tracks_dict = loaded_objects[0], min_track_length = 4, im_shape = loaded_objects[1], num_frames = loaded_objects[2], scale = 1, interpolate = False)
experiment_results = {"superres":superres, "velocity":velocity, "DensityIm_time":DensityIm_time,"im_shape":loaded_objects[1]}
savemat(ULMinfo['path'] + "experiment_results.mat", experiment_results)
savemat(ULMinfo['path'] + "experiment_info.mat", ULMinfo)

plt.figure(1)
plt.imshow(superres**(1/2),cmap = 'hot')
plt.savefig(ULMinfo['path'] + "superresolution.png")
plt.figure(2)
plt.imshow(velocity,cmap = plt.cm.nipy_spectral)
plt.savefig(ULMinfo['path'] + "velocitymap.png")
clb = plt.colorbar()
clb.ax.set_title('mm/sec')
plt.show()
