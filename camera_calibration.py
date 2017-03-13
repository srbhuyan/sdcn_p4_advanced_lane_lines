import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

class CameraCalibration:

    def __init__(self, chessboard_images_dir, pattern_size=(9, 6)):
        self.__chessboard_images_dir = chessboard_images_dir 
        self.__pattern_size          = pattern_size
        self.__dist_coeff            = None
        self.__camera_mtx            = None
        self.__camera_mtx_pickle     = "mtx_dist_pickle.p"

    def calibrate_camera(self, img_size):
        if img_size is None:
            print("No image size was passed")
            return

        objp       = np.zeros((self.__pattern_size[0] * self.__pattern_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.__pattern_size[0], 0:self.__pattern_size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(self.__chessboard_images_dir + '/*.jpg')

        print("Total Calibation Images = ", len(images))

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.__pattern_size, None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, self.__camera_mtx, self.__dist_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        dist_pickle = {}
        dist_pickle["mtx"] = self.__camera_mtx
        dist_pickle["dist"] = self.__dist_coeff
        pickle.dump(dist_pickle, open(self.__camera_mtx_pickle, "wb"))

    def undistort(self, img):
        if img is None:
            print("No image passed")
            return
        elif self.__camera_mtx is None:
            print("No Camera Matrix found: Camera is not calibrated - run 'calibrate_camera' first")
        elif self.__dist_coeff is None:
            print("No Distortion Coefficient found: Camera is not calibrated - run 'calibrate_camera' first")

        return cv2.undistort(img, self.__camera_mtx, self.__dist_coeff, None, self.__camera_mtx)

    def _read_camera_matrix_from_pickle(self):
        dist_pickle = pickle.load(open(self.__camera_mtx_pickle, "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        return mtx, dist

    def perspective_transform(self, img, src, dst):
        if img is None:
            print("No image passed")
            return
        elif self.__camera_mtx is None:
            print("No Camera Matrix found: Camera is not calibrated - run 'calibrate_camera' first")
            return
        elif self.__dist_coeff is None:
            print("No Distortion Coefficient found: Camera is not calibrated - run 'calibrate_camera' first")
            return

        undist = cv2.undistort(img, self.__camera_mtx, self.__dist_coeff, None, self.__camera_mtx)

        M    = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        img_size = (undist.shape[1], undist.shape[0])
        warped = cv2.warpPerspective(undist, M, img_size)
        return warped, M, Minv

