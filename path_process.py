import os
import copy

class PathProcess:
    def __init__(self,img_base):
        self.kind_list=os.listdir(img_base)
        self.kind_list=[kind for kind in self.kind_list if kind[0]!='.']
        self.file_name_list={}
        self.file_path_list={}
        for kind in self.kind_list:
            self.file_name_list[kind]=[
                file_name for file_name in os.listdir(img_base+'/'+kind)
                if file_name[-4:] == '.jpg'
            ]
            self.file_path_list[kind]=[
                img_base+'/'+kind+'/'+file_name
                for file_name in self.file_name_list[kind]
            ]


    def get_kind_list(self):
        return copy.deepcopy(self.kind_list)

    def get_file_name_list(self,kind):
        return copy.deepcopy(self.file_name_list[kind])

    def get_file_path_list(self,kind):
        return copy.deepcopy(self.file_path_list[kind])


def test():
    path=PathProcess('train_pic')
    print path.get_kind_list()
    print path.get_file_name_list('cloth')
    print path.get_file_path_list('cloth')

if __name__=='__main__':
    test()