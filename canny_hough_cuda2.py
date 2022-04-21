import cv2
import time
import numpy as np
import gpu_instance as gi

SLEEP_TIME = 0.0
do_canny = False


IMG_NAME = "test_img480.png"

THRESHOLD1 = 67
THRESHOLD2 = 110
APERATURE = 3

RHO = 1.9
THETA = 0.01837
THRESHOLD = 99

MODE = int(input("0: BOTH \n 1:CANNY ONLY \n 2:HOUGH \n") or "0")
do_canny = 0
do_hough = 0
if(MODE == 0):
    do_canny = 1
    do_hough = 1
elif MODE == 1: 
    do_canny = 1
else:
    do_hough = 1

NUM_RUNS = int(input("NUM RUNS: ") or "10")

IMG_NAME = str(input("INPUT IMAGE: ") or "test_img480.png")

def canny_hough_cuda(IMG_NAME, IMG_OUT, n_runs):

    gpu = gi.gpu_instance(rho=RHO, theta=THETA, hough_th=THRESHOLD, canny_threshold1=THRESHOLD1, canny_threshold2=THRESHOLD2, aperature=APERATURE)

    img = cv2.imread(IMG_NAME, 0)
    run_time_total = 0
    frame = None
    lines = None

    gpu_up_down_time_total = 0
    run_time_total = 0

    for i in range(n_runs):
        frame = img.copy()
        if frame.any():
            t_gpu_upload = time.perf_counter()
            gpu.upload(frame)
        
            t_cuda_canny_hough_start = time.perf_counter()
            if(do_canny):
                gpu.canny_edge()

            if(do_hough):    
                gpu.hough_lines()
            
            t_cuda_canny_hough_end = time.perf_counter()
        
            gpu.download()
            t_gpu_download = time.perf_counter()

            run_time = (t_gpu_download - t_gpu_upload) * 1000
            cuda_run_time = (t_cuda_canny_hough_end - t_cuda_canny_hough_start) * 1000
            gpu_up_and_down_time = run_time - cuda_run_time

            run_time_total += run_time
            gpu_up_down_time_total += gpu_up_and_down_time

            print(f"Run {i} in {round(run_time, 2)}ms taking {round(gpu_up_and_down_time)}ms for gpu upload + download")

            time.sleep(SLEEP_TIME)

    print(f"Avg {round(run_time_total/n_runs, 2)}ms over {n_runs} runs")
    print(f"Avg GPU upload + download {round(gpu_up_down_time_total/n_runs, 2)}")

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




canny_hough_cuda(IMG_NAME, "time.png", NUM_RUNS)