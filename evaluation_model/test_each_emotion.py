import keras
from keras.preprocessing import image
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications import resnet50
import sys
import os
import cv2
sys.path.append(os.getcwd())
from keras.applications.resnet50 import preprocess_input, decode_predictions
from utils_cv.image_processing_operations import *
from utils_cv.face_detection import *
from emotion_recognition.data_processing import create_emotion_vectors

emotion_code={}
emotion_code['Disgusted']=0
emotion_code['Happy']=1
emotion_code['Sad']=2
emotion_code['Surprise']=3

if __name__ == "__main__":
    emotions_list= os.listdir('dataset') 
    if '.DS_Store' in emotions_list:
        emotions_list.remove('.DS_Store')
        
    for emotion in emotions_list:
        x,y=create_emotion_vectors(emotion)
        np.save(os.getcwd()+'/evaluation_model/x_'+str(emotion),np.array(x))
        np.save(os.getcwd()+'/evaluation_model/y_'+str(emotion),np.array(y))

