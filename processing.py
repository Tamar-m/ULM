from scipy.io import loadmat
import numpy as np
import cv2
from skimage.measure import label, regionprops, regionprops_table
from scipy.signal import savgol_filter
from skimage.feature import peak_local_max

class Processing:
    def __init__(self):
        pass

    @staticmethod
    def loaddata(path,super_frame_num):
        """Load Image Data.

        Args: 
            path: folder containing data of type IQ or Image
            super_frame_num: number of superframe to load
        Returns:
            A 3D array containing all image frames in the supermat
        """
        # mat = loadmat(path+super_frame_num) 
        mat = loadmat(path+'%d'%(super_frame_num))
        if "ImageData_tot" in mat:
            return mat['ImageData_tot']
        elif "IQdata" in mat: # this was for the data from tali's paper
            return abs(mat['IQdata'])
        elif 'IData_tot' in mat:
            return np.hypot(mat['IData_tot'], mat['QData_tot'])

    @staticmethod
    def SVD_filt(super_frame,cutoff):
        """Perform SVD filter on the supermat.
        
        Args:
            super_frame: the supermat to filter
            cutoff: singular value cutoff to remove
        Returns:
            A SVD filtered array 
            """
        z, x, t = super_frame.shape
        cas = np.zeros((x*z,t))
        for i in range(0,t):
            col = np.reshape(super_frame[:,:,i],(x*z))
            cas[:,i] = col
        U, Sigma, VT = np.linalg.svd(cas,full_matrices=False)
        Sigma[0:cutoff] = 0
        Smat = np.diag(Sigma)
        cas_filtered = np.dot(np.dot(np.dot(cas,VT.T),VT),Smat)
        data_filtered = np.zeros(super_frame.shape)
        for i in range(0,t):
            im = np.reshape(cas_filtered[:,i],(z,x))
            data_filtered[:,:,i]=im

        return data_filtered

    @staticmethod
    def log_scale(im, db=1):
        """Convert an image to logscale between 0 and dB and rescale to 0-255
        No longer using for processing, only scaling between 0-255"""
        # im = (im - im.min()) / (im.max()-im.min())
        # b = 10**(-db/20)
        # a = 1-b 
        # im = 20 * np.log10(a * im + b) 
        im = np.uint8(255*((im-im.min())/(im.max()-im.min())))
        return im

    @staticmethod
    def weighted_av(im, coordinates):
        """Localize PSF center with weighted average
        
        Args: 
            im: preprocessed image 
            coordinates: coordinates of local maxima in im
        Returns:
            coords: the new coordinates localized using the weighted average of the intensity within the PSF shape"""
        mask = np.zeros(im.shape)
        mask[coordinates[:,0],coordinates[:,1]] = 1
        kernel = np.ones((2,8),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        intensity_im = im*mask
        props = regionprops_table(label_image = label(intensity_im>0), intensity_image=intensity_im, properties=['weighted_centroid'])
        coords = np.column_stack([props['weighted_centroid-0'],props['weighted_centroid-1']])
        return coords
    
    @staticmethod
    def interp(x):
        """Interpolate tracks
        Args:
            x: track to interpolate
        Returns:
            interpolated and smoothed track using savitzky-golay filter"""
        N = 300
        t, xp = np.linspace(0, 1, N), np.linspace(0, 1, len(x))
        return np.interp(t, xp, savgol_filter(x,min((len(x) - np.mod(len(x)-1,2)),5),2))
    

    @staticmethod
    def brightest_points(image, num_bubbles):
        """Localization using brightest points (may be better for in-vivo)
        Args:
            im: the frame to localize bubbles in
            num points: estimated number of bubbles per frame (try different options to optimize)
        Returns:
            coords: coordinates of localized bubbles"""
        coordinates = peak_local_max(image = image,min_distance=6,num_peaks = num_bubbles) 
        coords_localized = Processing.weighted_av(image,coordinates)
        return coords_localized
    
    @staticmethod
    def adaptive_thresh(image):
        """Localization using adaptive threshold 
        Args: 
            im: the frame to localize bubbles in
        Returns:
            coords: coordinates of localized bubbles"""
        
        filtered = cv2.bilateralFilter(image, d=7, sigmaColor=160, sigmaSpace=160)
        mask = cv2.adaptiveThreshold(filtered,255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,17,-12) # these are parameters you may need to play with to get good results
        filtered[mask==0] = 0
        # im_erode = cv2.erode(filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))) 
        # close small holes that are probably noise
        im_erode = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))) 
        regions, labels2 = cv2.connectedComponents(im_erode)
        output = np.zeros_like(image)

        for l in range(1, regions):  # Skip the background (label 0)
            region_mask = (labels2 == l)
            max_intensity = np.max(image[region_mask])
            mask_max_intensity = (image == max_intensity) * region_mask
            # Set all pixels in the output mask that do not have the maximum intensity to 0
            output[mask_max_intensity] = image[mask_max_intensity]

        peaks = np.zeros((filtered.shape[0]+2, filtered.shape[1]+2), np.uint8)
        labels = label(output)
        props = regionprops(labels, intensity_image=image)

        for obj in props:

            c = obj.weighted_centroid
            l, inp, peaks, bb = cv2.floodFill(np.float32(filtered), peaks, (int(c[1]),int(c[0])), newVal=255, loDiff=0, upDiff=30, flags=cv2.FLOODFILL_MASK_ONLY)
        
        peaks = peaks[1:-1, 1:-1]
        labels = label(peaks)
        peaks_list = regionprops_table(label_image = labels, intensity_image=image, properties=['weighted_centroid'])
        found_peaks = np.column_stack([peaks_list['weighted_centroid-0'],peaks_list['weighted_centroid-1']])

        return found_peaks



    


    