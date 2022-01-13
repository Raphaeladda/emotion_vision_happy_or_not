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
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input, decode_predictions
from utils_cv.image_processing_operations import *
from utils_cv.face_detection import *

#from google.colab import drive
#drive.mount('/content/drive')     
''' Importation de modules nécessaires à google collab'''


## Creation of the CNN:

#A partir d'un modèle déjà préentrainé à reconnaître certains objets: resnet 50
# On enlève la dernière couche



n_classes=4
resnet_model = resnet50.ResNet50(weights="imagenet",include_top=False,input_shape=(224, 224, 3))

model = Sequential()
model.add(resnet_model)

# On rajoute des couches finales

model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(Dropout(0.3))


model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))


model.add(Dense(n_classes, activation='softmax'))


# Modèle à entrainer:
model.compile(loss=keras.losses.categorical_crossentropy,optimizer="sgd")

#Entrainement du modèle:
# Recuperation des données

x=np.load(os.getcwd()+'emotion_recognition/x',np.array(x))   #x=np.load('/content/drive/My Drive/happy_or_not/x.npy')
y=np.load(os.getcwd()+'emotion_recognition/y',np.array(y))   #y=np.load('/content/drive/My Drive/happy_or_not/y.npy')


# Séparation en données d'entrainement et de test:
# 4/5 for training and 1/5 for testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = keras.utils.to_categorical(y_train)
test_Y_one_hot = keras.utils.to_categorical(y_test)

# Create validation set

x_train_final, x_val, y_train_final, y_val = train_test_split(x_train, train_Y_one_hot, test_size=0.2)

#Entrainement (réalisé sur Google collab avec des serveurs hébergés par Google, sous technologie GPU: beaucoup plus rapide)

model.fit(x_train_final,y_train_final,epochs=25,batch_size=64,verbose=1, validation_data=(x_val, y_val))

# Sauvegarde du modèle 
model.save(os.getcwd() +'emotion_recognition/neural_model_emotion.h5')      #model.save('/content/drive/My Drive/happy_or_notneural_model_emotion.h5')
                                                                            # Sur google collab
                                                                            # On le save afin de ne pas avoir à le réentrainer à chaque fois
                                                                            # Il suffit juste de le charger avec la fonction load_model
