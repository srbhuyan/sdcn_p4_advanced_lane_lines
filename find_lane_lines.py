import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def find(binary_warped, undist, Minv, left_line, right_line):
    
    if left_line.histogram_search == True:
        search, left_fitx, right_fitx, ploty, result_image = histogram_search(binary_warped, undist, Minv, left_line, right_line)
    else:
        search, left_fitx, right_fitx, ploty, result_image = targeted_search(binary_warped, undist, Minv, left_line, right_line)

    return search, left_fitx, right_fitx, ploty, result_image

def sanity_check(lc, rc):

    # Check curvature
    lc_log = math.log10(lc)
    rc_log = math.log10(rc)

    curvature_error_threshold = 1.1
    if math.fabs(lc_log - rc_log) > curvature_error_threshold:
        print("Sanity Check Failed: curvature")
        return False

    return True

def histogram_search(binary_warped, undist, Minv, left_line, right_line):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = out_img.astype(np.uint8)

    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Windows
    nwindows = 20
    window_height = np.int(binary_warped.shape[0]/nwindows)
 
    # Identify nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    # Windows +/- margin
    margin = 80
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds  = []
    right_lane_inds = []

    # Sliding windows
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Curvature
    left_curverad, right_curverad  = curvature(left_fitx, right_fitx, binary_warped.shape[0], ploty)
    mean_curvature = round( (left_curverad + right_curverad)/2 )

    # Visualize
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    '''
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    plit.savefig("output_images/test1.jpg")
    plt.show()
    '''

    search = np.copy(out_img)

    # Sanity Check
    sanity = sanity_check(left_curverad, right_curverad)

    if sanity == False:
        # If sanity check fails use previous fit
        left_fitx = left_line.last_fitx(ploty)
        right_fitx = right_line.last_fitx(ploty)

    # Draw to Raod
    result = draw_to_road(binary_warped, undist, Minv, left_fitx, right_fitx, ploty, mean_curvature)

    # Tracking
    if sanity == True:
        left_line.save_fit(left_fitx, ploty)
        right_line.save_fit(right_fitx, ploty)
 
        left_line.histogram_search = False

    return search, left_fitx, right_fitx, ploty, result
 
def targeted_search(binary_warped, undist, Minv, left_line, right_line):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    left_fit = left_line.best_fit(ploty)
    right_fit = right_line.best_fit(ploty)

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    # ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Curvature
    left_curverad, right_curverad = curvature(left_fitx, right_fitx, binary_warped.shape[0], ploty)
    mean_curvature = round( (left_curverad + right_curverad)/2 )

    # Visualize
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    '''
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    plt.savefig("output_images/test2.jpg")
    plt.show()
    '''

    search = np.copy(result)

    sanity = sanity_check(left_curverad, right_curverad)

    if sanity == False:
        left_fitx = left_line.last_fitx(ploty)
        right_fitx = right_line.last_fitx(ploty)

    # Draw to Road
    result = draw_to_road(binary_warped, undist, Minv, left_fitx, right_fitx, ploty, mean_curvature)

    # Tracking
    if sanity == True:
        left_line.save_fit(left_fitx, ploty)
        right_line.save_fit(right_fitx, ploty)

    left_line.sanity(sanity)

    return search, left_fitx, right_fitx, ploty, result

def curvature(left_fitx, right_fitx, y_eval, ploty):
    # Curvature
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def draw_to_road(binary_warped, undist, Minv, left_fitx, right_fitx, ploty, mean_curvature):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Draw Back To Road
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result)

    # Put radius of curvature on image 
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(mean_curvature)
    cv2.putText(result,text,(400,100), font, 1,(255,255,255),2)

    # Put distance from of the lane center
    pts = np.argwhere(newwarp[:,:,1])
    position = undist.shape[1]/2

    left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 700)][:,1])
    right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 700)][:,1])

    center = (left + right)/2
    dist_from_lane_center = (position - center) * xm_per_pix

    if dist_from_lane_center < 0:
        text = "Vehicle is {:.2f} m left of center".format(-dist_from_lane_center)
    else:
        text = "Vehicle is {:.2f} m right of center".format(dist_from_lane_center)
    cv2.putText(result,text,(400,150), font, 1,(255,255,255),2)

    #plt.show()

    return result
