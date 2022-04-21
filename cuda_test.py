import cv2
import time


gpu_frame = cv2.cuda_GpuMat()

def cannyBenchmarking(img_path):
    global gpu_frame
    frame = cv2.imread(img_path, 0)
    frame_normal = frame.copy()
    if frame.any():
        print("Running Canny Edge detection normal...")
        t_normal_canny_start = time.perf_counter()
        frame_normal = cv2.Canny(frame_normal, 67, 110, apertureSize=3)
        t_normal_canny_end = time.perf_counter()
        normal_run_time = (t_normal_canny_end - t_normal_canny_start) * 1000
        print(f"Canny Normal execution time{normal_run_time}ms")
        
        print("Running Canny Edge CUDA...")
        t_gpu_upload = time.perf_counter()
        gpu_frame.upload(frame)
    
        t_cuda_canny_start = time.perf_counter()
        det = cv2.cuda.createCannyEdgeDetector(low_thresh=67, high_thresh=110)
        det.setAppertureSize(3)
        frame = det.detect(gpu_frame)
        t_cuda_canny_end = time.perf_counter()
    
        frame.download()
        t_gpu_download = time.perf_counter()
    
        cuda_total_time = (t_gpu_download - t_gpu_upload) * 1000
        cuda_run_time = (t_cuda_canny_end - t_cuda_canny_start) * 1000
        gpu_up_and_down_time = cuda_total_time - cuda_run_time
        print(f"Cuda Canny Total Time: {cuda_total_time}ms")
        print(f"Cuda Canny run time: {cuda_run_time}ms")
        print(f"GPU Upload and Download Time: {gpu_up_and_down_time}ms")

def cannyHoughBenchmarking(img_path):
    global gpu_frame
    RHO = 1.9
    THETA = 0.01837
    THRESHOLD = 99

    frame = cv2.imread(img_path, 0)
    frame_normal = frame.copy()
    if frame.any():
        print("Running Canny and hough detection normal...")
        t_normal_canny_hough_start = time.perf_counter()
        frame_normal = cv2.Canny(frame_normal, 67, 110, apertureSize=3)
        frame_normal = cv2.HoughLines(frame_normal, RHO, THETA, THRESHOLD)
        t_normal_canny_hough_end = time.perf_counter()
        normal_run_time = (t_normal_canny_hough_end - t_normal_canny_hough_start) * 1000
        print(f"Canny Hough Normal execution time{normal_run_time}ms")
        
        print("Running Canny Edge and Hough CUDA...")
        t_gpu_upload = time.perf_counter()
        gpu_frame.upload(frame)
    
        t_cuda_canny_hough_start = time.perf_counter()
        canny_det = cv2.cuda.createCannyEdgeDetector(low_thresh=67, high_thresh=110)
        canny_det.setAppertureSize(3)
        canny_frame = canny_det.detect(gpu_frame)

        hough_det = cv2.cuda.createHoughLinesDetector(rho=RHO, theta=THETA, threshold=THRESHOLD) 
        hough_frame = hough_det.detect(canny_frame)
        t_cuda_canny_hough_end = time.perf_counter()
    
        hough_frame.download()
        t_gpu_download = time.perf_counter()
    
        cuda_total_time = (t_gpu_download - t_gpu_upload) * 1000
        cuda_run_time = (t_cuda_canny_hough_end - t_cuda_canny_hough_start) * 1000
        gpu_up_and_down_time = cuda_total_time - cuda_run_time
        print(f"Cuda Canny Hough Total Time: {cuda_total_time}ms")
        print(f"Cuda Canny Hough run time: {cuda_run_time}ms")
        print(f"GPU Upload and Download Time: {gpu_up_and_down_time}ms")


print("CANNY EDGE BENCHMARKS")
print("\n")
print("Testing on 480p Image")
cannyBenchmarking("test_img480.png")
print("\n")
print("Testing on 1080p Image")
cannyBenchmarking("test_img1080.png")
print("\n")
print("CANNY+HOUGH BENCHMARKS")
print("\n")
print("Testing on 480p Image")
cannyHoughBenchmarking("test_img480.png")
print("\n")
print("Testing on 1080p Image")
cannyHoughBenchmarking("test_img1080.png")
