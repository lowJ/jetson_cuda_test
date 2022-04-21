import cv2
class gpu_instance:

    def __init__(self, rho=1.9, theta=0.01837, hough_th=99, canny_threshold1=67, canny_threshold2=110, aperature=3):
        self.THRESHOLD1 = canny_threshold1
        self.THRESHOLD2 = canny_threshold2
        self.APERATURE = aperature

        self.RHO = rho
        self.THETA = theta
        self.THRESHOLD = hough_th

        self.canny_detector = cv2.cuda.createCannyEdgeDetector(low_thresh=self.THRESHOLD1, high_thresh=self.THRESHOLD2)
        self.canny_detector.setAppertureSize(self.APERATURE)

        self.hough_detector = cv2.cuda.createHoughLinesDetector(rho=self.RHO, theta=self.THETA, threshold=self.THRESHOLD) 

        self.gpu_frame = cv2.cuda_GpuMat()

    def upload(self, frame):
        self.gpu_frame.upload(frame)

    def download(self):
        return self.gpu_frame.download()

    def canny_edge(self):
        self.gpu_frame = self.canny_detector.detect(self.gpu_frame)

    def hough_lines(self):
        self.gpu_frame = self.hough_detector.detect(self.gpu_frame)
