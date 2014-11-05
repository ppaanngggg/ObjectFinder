from image_process import ImageProcess
import cv2

class ObjectProcess:
    def __init__(self,path,clf_fore,clf_shape):
        self.image=cv2.imread(path)
        self.image_proc_s=ImageProcess(self.image,70)
        self.image_proc_m=ImageProcess(self.image,100)
        self.image_proc_l=ImageProcess(self.image,130)