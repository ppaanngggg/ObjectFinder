import segmentation
import train
import os
import cv2
from pymongo import MongoClient


files=os.listdir('train_pic')
files=[filename for filename in files if filename[-4:]=='.jpg']
seg_list=[]
clf=train.train_sample('shape')

client=MongoClient()
db=client.object_finder
collection=db.bag_of_boundary

total=0

for filename in files:
    print filename
    image=cv2.imread('train_pic/'+filename)
    seg_m=segmentation.segmentation(image,100)
    seg_m.set_classify_target_list(clf.predict(seg_m.get_classify_vec_list()))
    seg_l=segmentation.segmentation(image,130)
    seg_l.set_classify_target_list(clf.predict(seg_l.get_classify_vec_list()))
    hog_list=seg_m.get_hog_list()+seg_l.get_hog_list()
    target_list=list(seg_m.get_classify_target_list())+list(seg_l.get_classify_target_list())
    for index in range(len(hog_list)):
        if target_list[index]:
            collection.insert(
                {
                    'hog':hog_list[index],
                    'weight':1./len(hog_list),
                    'file':filename
                }
            )
    total+=len(hog_list)

print '*** sum *** :',total