import cv2 as cv

frame = cv2.imread("test1.png")

gpu_frame = cv.cuda_GpuMat()

if frame:
    gpu_frame.upload(frame)

    frame = cv.cuda.resize(gpu_frame, (852, 480))
    frame.download()

    ret, frame = vod.read()
