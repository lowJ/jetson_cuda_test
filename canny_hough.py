import cv2
import time
import numpy as np

IMG_NAME = "test_img480.png"

THRESHOLD1 = 67
THRESHOLD2 = 110
APERATURE = 3

RHO = 1.9
THETA = 0.01837
THRESHOLD = 99

NUM_RUNS = int(input("NUM RUNS: ") or "10")

IMG_NAME = str(input("INPUT IMAGE: ") or "test_img480.png")

def canny_hough(IMG_NAME, IMG_OUT, n_runs):
    img = cv2.imread(IMG_NAME, 0)
    run_time_total = 0
    frame = None
    lines = None
    for i in range(n_runs):
        frame = img.copy()
        if frame.any():
            t_normal_canny_hough_start = time.perf_counter()
            frame = cv2.Canny(frame, THRESHOLD1, THRESHOLD2, apertureSize=APERATURE)
            lines = cv2.HoughLines(frame, RHO, THETA, THRESHOLD)
            t_normal_canny_hough_end = time.perf_counter()
            run_time = (t_normal_canny_hough_end - t_normal_canny_hough_start) * 1000
            run_time_total += run_time
            print(f"Run {i} in {round(run_time, 2)}ms")

    print(f"Avg {round(run_time_total/n_runs, 2)}ms over {n_runs} runs")

    # for line in lines:
    #     print("hi")
    #     rho,theta=line[0]
    #     a=np.cos(theta)
    #     b=np.sin(theta)
    #     x0=a*rho
    #     y0=b*rho
    #     x1 = int(x0+1000*(-b))
    #     y1 = int(y0+1000*(a))
    #     x2 = int(x0-1000*(-b))
    #     y2 = int(y0-1000*(a))
    #     cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),1)

    
    cv2.imwrite(IMG_OUT, frame)




canny_hough(IMG_NAME, "time.png", NUM_RUNS)


        

    



