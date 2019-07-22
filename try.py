# USAGE

'''
 python try.py  --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7


'''

# from imutils import paths

import argparse
# import imutils
import pickle
import cv2
import os
import time
from threading import *
import numpy as np
import threading


def led():
    os.system('python led1.py')
def song():
    os.system('mplayer chushihua.mp3') 
    os.system('mplayer qingtian.mp3')
def send():
   # image = imutils.resize(image, width=600)
   # (h, w) = image.shape[:2]

        # construct a blob from the image
   # ret, image = cam.read()   
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
           # if fW < 20 or fH < 20:
            #    continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            knownEmbedding =vec.flatten()
           # print(knownEmbedding)
            print('*****************************************************************')
            a = knownEmbedding.tolist()
            print(a)
            file = open('now.txt', 'w')
            file.write(str(a))
            file.close()
          #  time.sleep(3)
                        
    

if __name__ == '__main__':
    send_num=0
    t1 = threading.Thread(target=led)
    t2 = threading.Thread(target=song)
    
    t1.start()
    t2.start()
    
    cam = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    ap = argparse.ArgumentParser()

    #ap.add_argument("-d", "--detector", required=True,
    #   help="path to OpenCV's deep learning face detector")
    #ap.add_argument("-m", "--embedding-model", required=True,
    #   help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detection_model",
        "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")



    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    #os.system('mplayer chushihua.mp3')

    while(True):
        send_num = send_num+1

        ret, image = cam.read()
        
        
        AddText = image.copy()
        print(send_num)
        if(send_num%15 == 0):
        #t3.start()
       #     t3.join() 
       #    print('##############################################')
         
         send()

        gray = cv2.cvtColor(AddText, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(AddText, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(AddText,'people',(x+5,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            cv2.putText(AddText,'confidence',(x+5,y+h-5),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.imshow('image', AddText)
        
           
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
           break
        



    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


