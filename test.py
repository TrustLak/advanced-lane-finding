from helpers import *
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

###########################
# calibrate once
###########################
mtx, dist = calibrate()

###############################
# define pipeline to each image
###############################
def pipeline(img):
    # undistort example:
    
    
    # cv2.imshow('img', img)
    # cv2.waitKey(1000)
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #cv2.imshow('undist', undist)
    #cv2.waitKey(1000)
    
    # taken from slack discussion
    img_size = (img.shape[0], img.shape[1])
    
    src = np.float32([(570,470),(700,470),(250,660),(1030,660)])
    dst = np.float32([(450,0), (img_size[1]-450,0),(450,img_size[0]),(img_size[1]-450,img_size[0])])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, (img_size[1],img_size[0]), flags=cv2.INTER_LINEAR)
    # plt.imshow(warped)
    # plt.show()
    
    result = np.copy(warped)
    # print("warped shape:")
    # print(warped.shape)
    #cv2.imshow('warped', warped)
    # cv2.imwrite('warped.jpg', result)
    #cv2.waitKey(3000)
    
    
    
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    
    combined = apply_thresholds(result)
    
    binary_warped = combined
    # plt.imshow(binary_warped)
    # plt.show()
    
    #############################################################################################
    # Define method to fit polynomial to binary image with lines extracted, using sliding window
    #############################################################################################
    def sliding_window(img):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        quarter_point = np.int(midpoint//2)
        # Previously the left/right base was the max of the left/right half of the histogram
        # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
        leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
        rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint
        
        #print('base pts:', leftx_base, rightx_base)
    
        # Choose the number of sliding windows
        nwindows = 20
        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 60
        # Set minimum number of pixels found to recenter window
        minpix = 40
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Rectangle data for visualization
        rectangle_data = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
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
    
        left_fit, right_fit = (None, None)
        # Fit a second order polynomial to each
        if len(leftx) != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) != 0:
            right_fit = np.polyfit(righty, rightx, 2)
        
        visualization_data = (rectangle_data, histogram)
        
        return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data
    
    
    
    
    ###########################################
    # visualize data after sliding_window
    ###########################################
    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window(binary_warped)
    exampleImg = np.copy(img)
    exampleImg_bin = binary_warped
    h = exampleImg.shape[0]
    left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
    right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
    #print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)
    
    rectangles = visualization_data[0]
    histogram = visualization_data[1]
    
    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((exampleImg_bin, exampleImg_bin, exampleImg_bin))*255)
    # Generate x and y values for plotting
    ploty = np.linspace(0, exampleImg_bin.shape[0]-1, exampleImg_bin.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    for rect in rectangles:
    # Draw the windows on the visualization image
        cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
        cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = exampleImg_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    
    ###########################################################
    # draw lines:
    ###########################################################
    draw_img = draw_lane(img, binary_warped, left_fit, right_fit, Minv)


    ###########################################################
    # calculate radius and center position then draw data:
    ###########################################################
 # Define conversions in x and y from pixels space to meters
    # this section is adopted directly from slack discution
    y_scale = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    x_scale = 3.7/378  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_rad, right_rad, center = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = binary_warped.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)
  
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*y_scale, leftx*x_scale, 2)
        right_fit_cr = np.polyfit(righty*y_scale, rightx*x_scale, 2)
        # Calculate the new radii of curvature
        left_rad = ((1 + (2*left_fit_cr[0]*y_eval*y_scale + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_rad = ((1 + (2*right_fit_cr[0]*y_eval*y_scale + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
    
    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts 
    if right_fit is not None and left_fit is not None:
        car_position = binary_warped.shape[1]/2
        l_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        r_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center = (car_position - lane_center_position) * x_scale

    #print(center, left_rad, right_rad)
    average_radius = (left_rad + right_rad)/2
    #print(average_radius)

    # add data to plot:
    str1 = "R = " + str(round(average_radius, 2)) + "m"
    cv2.putText(draw_img, str1, (40,70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)
    str2 = "Vehicle is " + str(round(center, 2))+ "m left of center"
    cv2.putText(draw_img, str2, (40,110), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)

    return draw_img


mtx, dist = calibrate()

img = cv2.imread('./test_images/test6.jpg')
single_image_result = pipeline(img)
plt.imshow(single_image_result)
plt.show()



from moviepy.editor import VideoFileClip
full_result = 'project_video_labeled.mp4'
full_clip = VideoFileClip('project_video.mp4')
full_result_clip = full_clip.fl_image(pipeline)
full_result_clip.write_videofile(full_result, audio=False)

