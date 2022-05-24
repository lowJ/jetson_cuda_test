import cv2
from cv2 import cuda
import time
import numpy as np
import gpu_instance as gi
import math


VID_NAME = "road_vid480.mov"
#VID_NAME = "sample.mp4"

NUM_RUNS = 5


#https://stackoverflow.com/questions/41098237/is-the-warmup-code-necessary-when-measuring-cuda-kernel-running-time
#First few runs allow GPU to warm up


#VID_NAME = str(input("INPUT VID: ") or "road_vid480.mov")

def rectTopDownMask(img, h):
    height = img.shape[0]
    width = img.shape[1]
    #rect = np.array([[(0, 0), (width, 0), (0, rectHeight), (width, rectHeight)]])
    rect = np.array([[(0, height), (0, height - h), (width, height - h), (width, height)]])
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, rect, 255)
    mask = cv2.bitwise_and(img, mask)
    return mask

def applyGaussianBlur(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def getCannyEdges(img, thresh1=55, thresh2=95): #TODO: parameterize
    return cv2.Canny(img, thresh1, thresh2, apertureSize=3)

def getHoughlines(img, resolutionRho, resolutionTheta, thresh):
    return cv2.HoughLines(img, resolutionRho, resolutionTheta, thresh)

def applyGscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def applyContrastAdjustment(img, a, b, doContrastThreshold):
    mean = cv2.mean(img)
    imgMean = (mean[0] + mean[1] + mean[2]) / 3;

    # contrast increase for low light
    grey = None
    if (imgMean < doContrastThreshold):
        img = cv2.convertScaleAbs(img, alpha=a, beta=b)

    return img

def convertPolarToCartesianGPU(lines):
    #print("ur mom")
    #print(lines)
    rtnLines = []
    for line in lines:
        len = 2000
        rho,theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + len*(-b))
        y1 = int(y0 + len*(a))
        x2 = int(x0 - len*(-b))
        y2 = int(y0 - len*(a))
        rtnLines.append((x1, y1, x2, y2))
        #cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    return rtnLines

def drawAverageLines(img, avg):
    if(np.any(avg[0])):
        cv2.line(img, (avg[0][0], avg[0][1]), (avg[0][2], avg[0][3]), (0, 255, 0), 5)
    if(np.any(avg[1])):
        cv2.line(img, (avg[1][0], avg[1][1]), (avg[1][2], avg[1][3]), (0, 255, 0), 5)


def rectTopDownCrop(img, h):
    height = img.shape[0]
    width = img.shape[1]
    cropped = img[height-h:height, 0:width]
    return cropped 

def drawMidPointLine(img, mid):
    #TODO do not hardcode image dimmensions
    # h = 480
    # w = 640

# height, width, number of channels in image
    h = img.shape[0]
    w = img.shape[1]
    mid = int(mid)
    mid_line_length = 40
    cv2.line(img, (mid, h), (mid, h-mid_line_length), (0, 150, 255), 8)
    cv2.line(img, (int(w/2), h), (int(w/2), (h-mid_line_length) - 100), (255, 150, 255), 4)
def pushingPLoop(mid):
    #hardcoded mid offset
    mid_offset = 0
    mid = mid + mid_offset

    steering_range = 180

    mid_max = 320+100
    mid_min = 320-100

    if(mid > mid_max):
        mid = mid_max

    if(mid < mid_min):
        mid = mid_min


    mid_zeroed = mid - mid_min
    pushing_p = steering_range/(mid_max - mid_min)


    angle = pushing_p * mid_zeroed

    #reflect on y axis bc that is what ros steering uses
    angle = 180 - angle

    return angle


def average(img, lines):

    def make_points(img, average):
        #print("average" + str(average))
        slope, y_int = average
        y1 = img.shape[0]
        y2 = int(y1 * (1/4))
        x1 = int((y1 - y_int) // slope )
        x2 = int((y2 - y_int) // slope )
        return np.array([x1, y1, x2, y2])

    slopes = []
    for line in lines:
        x1, y1, x2, y2  = np.reshape(line, 4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        slopes.append((slope, y_int))

    # filter horizontal slopes
    HORIZONTAL_THRESHOLD = 0.00001
    slopes = [s for s in slopes if s[0] > HORIZONTAL_THRESHOLD or s[0] < -HORIZONTAL_THRESHOLD]

    left = [s for s in slopes if s[0] < 0]
    right = [s for s in slopes if s[0] > 0]
    left_line = None
    right_line = None
    if(len(left)):
        left_avg = np.average(left, axis=0)
        left_line = make_points(img, left_avg)

    if(len(right)):
        right_avg = np.average(right, axis=0)
        right_line = make_points(img, right_avg)

    return [left_line, right_line]

def getXMidPointFromAvgLines(img, avg_lines):
    #y2-y1/x2-x1

    left = avg_lines[0]
    right = avg_lines[1]
    m_left = (left[3] - left[1])/(left[2]-left[0])
    m_right = (right[3] - right[1])/(right[2]-right[0])

    #TODO: idk why i had to do this but in desmos this looked right
    #img_h =  480#TODO: do not hardcode
    img_h = img.shape[0]


    m_left = m_left * -1
    m_right = m_right * -1

    #Sample x values for y=edge of image
    #TODO: do not hardcode
    xLeftAtBottomEdge = ((img_h - left[1])/m_left) + left[0]
    xRightAtBottomEdge = ((img_h - right[1])/m_right) + right[0]

    mid = ((xRightAtBottomEdge - xLeftAtBottomEdge)/2) + xLeftAtBottomEdge

    offset = 0

    mid = mid + offset

    return mid

def findSteeringAngle(img, avg_lines, drawMidLines=False):
    # parameters needed to find offset: image height, width, and horizontal mid point
    height = img.shape[0]
    width = img.shape[1]
    mid = int(width/2)

    # compute x_offset, or deviation from center line
    # if x_offset < 0, steer left and if x_offset > 0, steer right

    # TWO LINE CASE, both avg_lines[0] and avg_lines[1] (left and right) exist
    if(np.any(avg_lines[0]) and np.any(avg_lines[1])):
        #Kellys Logic
        # left_x2 = avg_lines[0][2]
        # right_x2 = avg_lines[1][2]
        # x_offset = int((left_x2 + right_x2) / 2 - mid)
        # y_offset = int(height/2)

        #Jonathan Edit
        mid = getXMidPointFromAvgLines(img, avg_lines=avg_lines)
        mid = int(mid)
        if drawMidLines and mid != None:
           drawMidPointLine(img, mid)

        angle = pushingPLoop(mid)
        return [angle, mid]



    # ONE LINE CASE
    elif(np.any(avg_lines[0]) and not np.any(avg_lines[1])):
        x1 = avg_lines[0][0]
        x2 = avg_lines[0][2]
        x_offset = x2 - x1
        y_offset = int(height/2)
    elif(np.any(avg_lines[1]) and not np.any(avg_lines[0])):
        x1 = avg_lines[1][0]
        x2 = avg_lines[1][2]
        x_offset = x2 - x1
        y_offset = int(height/2)

    # if no lines detected, return nothing
    else:
        return None
        print("No Lines Detected")

    # compute angle between heading vector and center line
    angle_to_mid_rad = np.arctan(x_offset/y_offset)
    angle_to_mid_deg = int(angle_to_mid_rad * 180 / np.pi)

    # compute steering angle: input to servo to steer based on lane lines
    steering_angle = 90 - angle_to_mid_deg

    return [steering_angle, None]

def displayHeadingLine(img, steering_angle):
    if(steering_angle == None):
        return img
    #Temp fix to handle divide by 0 error when steering angle is 180
    if(steering_angle >= 180):
        steering_angle = 179
    if(steering_angle <= 0):
        steering_angle = 1
    else:
        height = img.shape[0]
        width = img.shape[1]

    #    steering_angle = 180 - steering_angle
        # compute angle between heading vector and center line 
        angle_to_mid_deg = 90 - steering_angle

        # compute steering angle for heading vector display 
        steering_angle_display = angle_to_mid_deg + 90
        steering_angle_rad = steering_angle_display / 180 * np.pi

        # compute points needed to display heading vector line 
        x1 = int(width/2)
        y1 = height
        x2 = int(x1 - height / 2 / np.tan(steering_angle_rad))
        y2 = int(height/2)

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, "Angle: " + str(round(steering_angle, 2)), (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)



def shift_lane_lines_down(frame, avg, y):
    height = frame.shape[0]
    width = frame.shape[1]
    y = height-y
    if(np.any(avg[0])):
        avg[0][1] += y
        avg[0][3] += y

    if(np.any(avg[1])):
        avg[1][1] += y
        avg[1][3] += y
    return avg

def canny_hough_cuda(VID_NAME, IMG_OUT, n_runs):
    alpha=0.6
    beta=0.4
    contrastAdjustThreshold=127
    resolutionRho=1.9
    resolutionTheta=0.01837 #more processing thoq
    threshold=99
    ksize=7
    cannyFirstThreshold=57
    cannySecondThreshold=144
    rectTopMaskHeight=90
    rectBottomMaskHeight=640


    fps_avg_over_runs = 0 
    for i in range(n_runs):
        vid = cv2.VideoCapture(VID_NAME)
        frame_count = 0
        fps_avg = 0

        t_start = time.perf_counter()
        gpu_frame=cv2.cuda_GpuMat()
        cannyDetector = cv2.cuda.createCannyEdgeDetector(low_thresh=100, high_thresh=200)

        detectorHough = cv2.cuda.createHoughLinesDetector(rho=resolutionRho, theta=resolutionTheta, threshold=threshold)

        while(1):
            ret, frame = vid.read()
            #Create Detecor Objects for using algorithims in gpu

            if ret:
                if frame.any():
                    frame_count += 1

                    #Save a copy of the original frame for streaming later
                    orig_frame = frame.copy()

                    start = time.time()

                    #Blur to remove noise
                    frame = applyGaussianBlur(frame, ksize)

                    #Convert to Grey scale
                    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    #Crop the region of interest
                    frame = rectTopDownCrop(frame, rectTopMaskHeight)
                    
                    #Upload frame to GPU
                    gpu_frame.upload(frame)
 
                    #Run Canny Edge detection on GPU frame
                    canny_frame = cannyDetector.detect(gpu_frame)

                    #Run Hough Edge detection and return array of lines
                    lines = detectorHough.detect(canny_frame)

                    #Download array of lines from GPU
                    hough_gpu_lines = lines.download()

                    #Extract Relevent Data from the array
                    hough_gpu_lines = hough_gpu_lines[0]

                    #Convert the lines to cartesian form
                    lines_gpu = convertPolarToCartesianGPU(hough_gpu_lines)

                    #Parse the array and average out lines to get 2 distinct lane lines
                    lanes_gpu = average(frame, lines_gpu)

                    #Lane detection was done on cropped frame, so
                    #shift the lane lines down so they can be drawn on the original frame properly
                    lanes_gpu = shift_lane_lines_down(orig_frame, lanes_gpu, rectTopMaskHeight)

                    #Caclulate the steering angle
                    data = findSteeringAngle(orig_frame, lanes_gpu, drawMidLines=True)
                    elapsed = time.time() - start
                    elapsed = elapsed * 1000
 
                    fps = 1000/elapsed
                    fps_avg += fps

                    cv2.putText(orig_frame, "fps: " + str(round(fps, 1)), (0, 75), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


                    #Draw the lane lines on the original frame
                    drawAverageLines(orig_frame, lanes_gpu)

                    #Draw the steering angle line
                    displayHeadingLine(orig_frame, data[0])


                    #Show the original frame
                    cv2.imshow('LaneDetect', orig_frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            else:
                break

        print("Average FPS:")
        avg = fps_avg/frame_count
        print(avg)
        fps_avg_over_runs += avg



    print(f"AVG {fps_avg_over_runs/n_runs} FPS for {n_runs} RUNS")

canny_hough_cuda(VID_NAME, "time.png", NUM_RUNS)
