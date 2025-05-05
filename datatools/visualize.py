import os
import numpy as np
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import json
import cv2
#from ultralytics.data.utils import visualize_image_annotations


# from: https://github.com/jamesjg/FoodInsSeg/blob/main/visualize.py
if __name__=='__main__':

    
    # Set paths and directories
    json_file = r'anotations_coco\\foodins_train.json'
    imgsPath = r'datasets\\full\\images\\train'
    savePath = r'true_masks\\train'
    os.makedirs(savePath, exist_ok=True)

    jsonFile = json.load(open(json_file))
    classId2className = jsonFile['categories']
    imId2imageNames = {}

    # Map img position to image file name, e.g. 0 -> 000048.jpg
    for imgs in jsonFile['images']:
        imId2imageNames[imgs['id']] = imgs['file_name']
    imgId2labels = {}

    for label in jsonFile['annotations']:
        imgId = label['image_id']
        if imgId not in imgId2labels:
            imgId2labels[imgId] = []
        imgId2labels[imgId].append(label)

    #print(imgId2labels[0])
    #raise ImportError
    coco = COCO(json_file)

    for imgId in sorted(imgId2labels.items(), key=lambda x:int(x[0])):
        imgName = imId2imageNames[imgId[0]]
        img = io.imread(os.path.join(imgsPath,imgName))

        plt.figure(imgName)
        plt.axis('off')
        plt.imshow(img) 
        coco.showAnns(imgId2labels[imgId[0]])

        #print(imgName)
        #plt.show()  
        saveFilePath = os.path.join(savePath, f'{os.path.splitext(imgName)[0]}_mask.png')
        plt.savefig(saveFilePath, bbox_inches='tight', pad_inches=0)
        plt.close()
    