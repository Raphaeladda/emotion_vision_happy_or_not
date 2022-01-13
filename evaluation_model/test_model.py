import keras
from sklearn.metrics import accuracy_score
from keras.preprocessing import image
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications import resnet50
import sys
import os
sys.path.append(os.getcwd())
import cv2
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input, decode_predictions
from utils_cv.image_processing_operations import *
from utils_cv.face_detection import *

model=load_model(os.getcwd()+'/emotion_recognition/neural_model_emotion.h5') 

x_Happy=np.load(os.getcwd()+'/emotion_recognition/x_Happy.npy')
y_Happy=np.load(os.getcwd()+'/emotion_recognition/y_Happy.npy')
x_Sad=np.load(os.getcwd()+'/emotion_recognition/x_Sad.npy')
y_Sad=np.load(os.getcwd()+'/emotion_recognition/y_Sad.npy')
x_Surprise=np.load(os.getcwd()+'/emotion_recognition/x_Surprise.npy')
y_Surprise=np.load(os.getcwd()+'/emotion_recognition/y_Surprise.npy')
x_Disgusted=np.load(os.getcwd()+'/emotion_recognition/x_Disgusted.npy')
y_Disgusted=np.load(os.getcwd()+'/emotion_recognition/y_Disgusted.npy')


# 4/5 for training and 1/5 for testing

x_train_Happy, x_test_Happy, y_train_Happy, y_test_Happy = train_test_split(x_Happy, y_Happy, test_size=0.1)
x_train_Sad, x_test_Sad, y_train_Sad, y_test_Sad = train_test_split(x_Sad, y_Sad, test_size=0.1)
x_train_Disgusted, x_test_Disgusted, y_train_Disgusted, y_test_Disgusted = train_test_split(x_Disgusted, y_Disgusted, test_size=0.1)
x_train_Surprise, x_test_Surprise, y_train_Surprise, y_test_Surprise = train_test_split(x_Surprise, y_Surprise, test_size=0.1)

x_train=np.concatenate((x_train_Happy,x_train_Sad,x_train_Disgusted,x_train_Surprise))
y_train=np.concatenate((y_train_Happy,y_train_Sad,y_train_Disgusted,y_train_Surprise))

test_Y_one_hot_Happy = keras.utils.to_categorical(y_test_Happy,num_classes=4)
test_Y_one_hot_Sad = keras.utils.to_categorical(y_test_Sad,num_classes=4)
test_Y_one_hot_Disgusted = keras.utils.to_categorical(y_test_Disgusted,num_classes=4)
test_Y_one_hot_Surprise = keras.utils.to_categorical(y_test_Surprise,num_classes=4)

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = keras.utils.to_categorical(y_train)

x_test=np.concatenate((x_test_Happy,x_test_Sad,x_test_Disgusted,x_test_Surprise))
y_test=np.concatenate((y_test_Happy,y_test_Sad,y_test_Disgusted,y_test_Surprise))

# Change the labels from categorical to one-hot encoding
test_Y_one_hot = keras.utils.to_categorical(y_test)

# Create validation set

x_train_final, x_val, y_train_final, y_val = train_test_split(x_train, train_Y_one_hot, test_size=0.2)

# Calcul du taux de réussite 

def accuracy(x_test,test_Y_one_hot):
  ''' Prend en argument les données à tester:
  - x_test
  - test_Y_one_hot
  La fonction fait des prédictions à partir de x_test et compare avec les résultats que l'on devrait obtenir (tes_y_one_hot)
  Elle calcule donc l'accuracy'''

  y_pred=model.predict(x_test)
  n,m=y_pred.shape
  for i in range(n):
    index_max=np.argmax(y_pred[i])
    for j in range(m):
      if j==index_max:
        y_pred[i][j]=1
      else:
        y_pred[i][j]=0
  y_pred=np.array(y_pred,dtype='float32')
  return(accuracy_score(test_Y_one_hot, y_pred))


''' On évalue ici l'accuracy pour chacune des émotions'''

print("Happy accuracy: ",accuracy(x_test_Happy,test_Y_one_hot_Happy))
print("Sad accuracy: ",accuracy(x_test_Sad,test_Y_one_hot_Sad))
print("Surprise accuracy: ",accuracy(x_test_Surprise,test_Y_one_hot_Surprise))
print("Disgusted accuracy: ",accuracy(x_test_Disgusted,test_Y_one_hot_Disgusted))

print("Total accuracy of the model : ",accuracy(x_test,test_Y_one_hot))