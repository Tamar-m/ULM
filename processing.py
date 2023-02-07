from scipy.io import loadmat
import numpy as np
import cv2
from skimage.measure import label, regionprops

class Processing:
    def __init__(self):
        pass

    @staticmethod
    def loaddata(path,super_frame_num):
        """Load Image Data.

        Args: 
            path: folder containing data
            super_frame_num: number of superframe to load
        Returns:
            A 3D array containing all image frames in the supermat
        """
        mat = loadmat(path+'%d'%(super_frame_num))
        return mat['ImageData_tot']

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
        cas_filtered = np.dot(U * Sigma, VT)
        data_filtered = np.zeros(super_frame.shape)
        for i in range(0,t):
            im = np.reshape(cas_filtered[:,i],(z,x))
            data_filtered[:,:,i]=im
        return data_filtered

    @staticmethod
    def log_scale(im, db=1):
        """Convert an image to logscale between 0 and dB and rescale to 0-255"""
        im = (im - im.min()) / (im.max()-im.min())
        b = 10**(-db/20)
        a = 1-b 
        im = 20 * np.log10(a * im + b) 
        im = np.uint8(255*((im-im.min())/(im.max()-im.min())))
        return im

    @staticmethod
    def weighted_av(im, coordinates, fwhmy,fwhmx,fovy,fovx):
        """Localize PSF center with weighted average
        
        Args: 
            im: preprocessed image 
            coordinates: coordinates of local maxima in im
            fwhmy: the full width half max of the PSF of the system in the y direction [mm]
            fwhmx: the full width half max of the PSF of the system in the x direction [mm]
            fovy: the field of view of the image in the y direction [mm]
            fovx: the field of view of the image in the x direciton [mm]
        Returns:
            coords: the new coordinates localized using the weighted average of the intensity within the PSF shape"""
        mask = np.zeros(im.shape)
        mask[coordinates[:,0],coordinates[:,1]] = 1
        ppmx = ((fovx[1]-fovx[0]))/im.shape[1]
        ppmy = ((fovy[1]-fovy[0]))/im.shape[0]
        kernel = np.ones((np.rint(fwhmy/ppmy).astype(int),np.rint(fwhmx/ppmx).astype(int)),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        intensity_im = im*mask
        props = regionprops(label(intensity_im>0), intensity_image=intensity_im)
        coords = []
        for obj in props:
            c = obj.weighted_centroid
            coords.append(np.array([c[0],c[1]]))
        return np.array(coords)

    