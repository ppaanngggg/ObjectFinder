import os
import copy

class PathProcess:
    def __init__(self,img_base):
        self.kind_list=os.listdir(img_base)


    def get_kind_list(self):
        return copy.deepcopy(self.kind_list)