import cv2
from cv2 import cuda
import time
import numpy as np
import gpu_instance as gi
import math


VID_NAME = "road_vid480.mov"
#VID_NAME = "sample.mp4"

NUM_RUNS = 1


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
    #gpu = gi.gpu_instance(rho=RHO, theta=THETA, hough_th=THRESHOLD, canny_threshold1=THRESHOLD1, canny_threshold2=THRESHOLD2, aperature=APERATURE)

    run_time_total = 0
    frame = None
    lines = None

    gpu_up_down_time_total = 0
    run_time_total = 0
    fps_accumulate = 0

    for i in range(n_runs):
        vid = cv2.VideoCapture(VID_NAME)
        frame_count = 0

        t_start = time.perf_counter()
        gpu_frame=cv2.cuda_GpuMat()

        while(1):
            ret, frame = vid.read()
            if ret:
                if frame.any():
                    frame_count += 1
                    orig_frame = frame.copy()
                    frame = applyGaussianBlur(frame, ksize)
                    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    frame = cv2.Canny(frame, cannyFirstThreshold, cannySecondThreshold, apertureSize=3)

                    frame = rectTopDownMask(frame, rectTopMaskHeight)
                    
                    #Do cuda stuff here

                    #Allocate memory on gpu
                    #tempLines = cv2.cuda_GpuMat()

                    #upload the current frame to the gpu



                    hough_cpu_lines = cv2.HoughLines(frame,resolutionRho,resolutionTheta,threshold)
                    gpu_frame.upload(frame)
                    #cuMat = cuMat.convertTo(cv2.CV_8UC1, cuMat)

                    #create the canny detector object (way may only need to call this once?)
                    #detector = cv2.cuda.createCannyEdgeDetector(cannyFirstThreshold, cannySecondThreshold)
                    cannyDetector = cv2.cuda.createCannyEdgeDetector(low_thresh=100, high_thresh=200)

                    #run the canny edge algorithm on the frame in the gpu, cuMat contains the result

                    dstImg = cannyDetector.detect(gpu_frame)
                    


                    #img_cu = img_cu.converTo(cv2.CV_32FC1, img_cu)

                    ##cv2.HoughLines(edges,resolutionRho,resolutionTheta,threshold)
                    ##Create the hough lines detector (replacing function as seen above)
                    detectorHough = cv2.cuda.createHoughLinesDetector(rho=resolutionRho, theta=resolutionTheta, threshold=threshold)

                    ##Run the algorithim and detect lines, place detected lines into gpu memory (tempLines), the cuMat does not get modified
                    lines = detectorHough.detect(dstImg)


                    ##Download the lines in the gpu memory (tempLines) into the cpu memory (res)
                    res = None

                    #Download the cuMat, this shold just return canny edge applied 
                    #newframe = cuMat.download()
#                    len(res)
#                    len(res[0])
                    hough_gpu_lines = lines.download()
                    hough_gpu_lines = hough_gpu_lines[0]

                    #hough_gpu_lines = sorted(hough_gpu_lines, key= lambda x:x[1])

                    print("GPU LINES")
                    #for x in hough_gpu_lines:
                    #    print(x)

                    print("CPU LINES")
                    #hough_cpu_lines = sorted(hough_cpu_lines, key= lambda x:x[0][1])
                    #print(hough_cpu_lines)
                    #for x in hough_cpu_lines:
                    #    print(x)
                    #print(hough_cpu_lines)


                    lines_gpu = convertPolarToCartesianGPU(hough_gpu_lines)
                    #print(lines_gpu)

                    lanes_gpu = average(frame, lines_gpu)
                    print(lanes_gpu)

                    drawAverageLines(orig_frame, lanes_gpu)

                    


                    cv2.imshow('hi', orig_frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break


canny_hough_cuda(VID_NAME, "time.png", NUM_RUNS)
