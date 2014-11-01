import segmentation
import train
import os
import cv2
from pymongo import MongoClient


files=os.listdir('train_pic')
files=[filename for filename in files if filename[-4:]=='.jpg']
seg_list=[]
clf=train.train_sample('fore')
orb=cv2.ORB_create()

client=MongoClient()
db=client.object_finder
collection=db.bag_of_visual_word

total=0

for filename in files:
    print filename
    image=cv2.imread('train_pic/'+filename)
    seg=segmentation.segmentation(image,70)
    seg.set_classify_target_list(clf.predict(seg.get_classify_vec_list()))
    seg.compute_foreground_mask()
    # seg.compute_foreground_image()
    kp_list=orb.detect(seg.get_image(),seg.get_foreground_mask())
    kp_list,des_list=orb.compute(seg.get_image(),kp_list)
    if len(kp_list):
        total+=len(kp_list)
        # tmp=cv2.drawKeypoints(seg.get_foreground_image(),kp_list,None)
        # cv2.imshow('tmp',tmp)
        # cv2.waitKey()
        for des in des_list:
            collection.insert(
                {'des':[int(num) for num in list(des)],
                 'weight':1./len(kp_list),'file':filename}
            )

print '*** sum *** :',total