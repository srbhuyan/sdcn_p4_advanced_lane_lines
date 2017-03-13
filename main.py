import cv2
import numpy as np
import glob
import math
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import find_lane_lines
from line import Line
from image_processor import ImageProcessor
from camera_calibration import CameraCalibration

def color_gradient_threshold_pipeline(image, img_processor):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]

    ksize = 7

    gradx      = img_processor.abs_sobel_thresh (gray, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady      = img_processor.abs_sobel_thresh (gray, orient='y', sobel_kernel=ksize, thresh=(60, 255))
    mag_binary = img_processor.mag_thresh       (gray,             sobel_kernel=ksize, thresh=(40, 255))
    dir_binary = img_processor.dir_threshold    (gray,             sobel_kernel=ksize, thresh=(0.65, 1.05))

    grad_combined = np.zeros_like(dir_binary)
    grad_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    s_binary = np.zeros_like(grad_combined)
    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1


    grad_color_combined = np.zeros_like(grad_combined)
    grad_color_combined[(s_binary == 1) | (grad_combined == 1)] = 1

    # Stacking contributions from gradient and color  
    grad_color_stack = np.dstack(( np.zeros_like(grad_color_combined), grad_combined, s_binary))

    return grad_color_stack, grad_color_combined


def image_pipeline(image):

    grad_color_stack, img = color_gradient_threshold_pipeline(image, img_processor)
    birds_eye, M, Minv = calibration.perspective_transform(img, src, dst)
    undist = calibration.undistort(image)
    search, left_fitx, right_fitx, ploty, image_final = find_lane_lines.find(birds_eye, undist, Minv, left_line, right_line)

    scipy.misc.imsave('output_images/undist.jpg', undist)

    return grad_color_stack, img, birds_eye, search, left_fitx, right_fitx, ploty, image_final

def video_pipeline(image):

    grad_color_stack, img = color_gradient_threshold_pipeline(image, img_processor)
    birds_eye, M, Minv = calibration.perspective_transform(img, src, dst)
    undist = calibration.undistort(image)
    search, left_fitx, right_fitx, ploty, image_final = find_lane_lines.find(birds_eye, undist, Minv, left_line, right_line)

    return image_final


#__main__

# Camera Calibration
img = mpimg.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

calibration = CameraCalibration(chessboard_images_dir='camera_cal', pattern_size=(9,6))
calibration.calibrate_camera(img_size)

# Image Processor
img_processor = ImageProcessor()

# src and dst for perspective transform
test_img = mpimg.imread('test_images/straight_lines2.jpg')
test_img_size = (test_img.shape[1], test_img.shape[0])

src = np.float32([[580,460], [710,460], [1150,720], [150,720]])
offset = 200
dst = np.float32([ [offset, 0],
                   [test_img_size[0]-offset, 0],
                   [test_img_size[0]-offset, test_img_size[1]-0],
                   [offset, test_img_size[1]-0]
                 ])

video = True

left_line  = Line()
right_line = Line()

# Image processing
if video == False:
    images = glob.glob('test_images/*.jpg') 

    for idx, fname in enumerate(images):
        img_raw = mpimg.imread(fname)

        grad_color_stack, grad_color, birds_eye, search, left_fitx, right_fitx, ploty, img_final = image_pipeline(img_raw)
        
        '''
        scipy.misc.imsave('output_images/grad_color_stack.jpg', grad_color_stack)
        scipy.misc.imsave('output_images/binary.jpg', grad_color)
        scipy.misc.imsave('output_images/birds_eye.jpg', birds_eye)
        scipy.misc.imsave('output_images/search.jpg', search)
        scipy.misc.imsave('output_images/image_final.jpg', img_final)
        '''

        f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(36, 9))
        f.tight_layout()

        ax1.set_title('Original', fontsize=20)
        ax1.imshow(img_raw)
        ax2.set_title('Grad-Color-Stack', fontsize=20)
        ax2.imshow(grad_color_stack, cmap='gray')
        ax3.set_title('Gradient', fontsize=20)
        ax3.imshow(grad_color, cmap='gray')
        ax4.set_title('Birds Eye', fontsize=20)
        ax4.imshow(birds_eye, cmap='gray')
        ax5.set_title('Search', fontsize=20)
        ax5.plot(left_fitx, ploty, color='yellow')
        ax5.plot(right_fitx, ploty, color='yellow')
        ax5.imshow(search)
        ax6.set_title('Final', fontsize=20)
        ax6.imshow(img_final)

        plt.show()

# video processing
if video == True:
    white_output = 'project_video_out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(video_pipeline)
    white_clip.write_videofile(white_output, audio=False)

