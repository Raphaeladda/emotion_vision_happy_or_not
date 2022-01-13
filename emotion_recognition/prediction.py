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
from keras.models import load_model


emotion_code={}
emotion_code['Disgusted']=0
emotion_code['Happy']=1
emotion_code['Sad']=2
emotion_code['Surprised']=3

def processing_face_for_model(face_img): #This process only one face
    '''Dimensionnement des images afin qu'elles puissent entrer dans notre réseau de neurones
       Format (1,224,224,3)'''
    img = resize_image(face_img,(224,224))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.array([img])
    img = np.stack((img,)*3,axis=-1)
    return(img)

def process_for_model(img):
    face_list = face_detect(img)
    res = []
    for face in face_list:
        res.append(processing_face_for_model(face))
    return(res)

def predict(img,cnn):
    '''Prend en argument le réseau de neurone et l'image à prédire et renvoie un vecteur de taille (1,4), 
    ou chaque élément est la probabilité correspondant à chaque émotion:
    position 0: Disgusted
    position 1: Happy
    position 2: Sad
    position 3: Surprise'''
    return(cnn.predict(img))

def decode_predictions(prediction):
    ''' Prend en argument un vecteur de taille 4, qui est un vecteur de prédiction et
      renvoie l'émotion pour laquelle la probabilité est le plus élevée, ainsi que la probabilité associée
     (c'est donc un tuple)
    
    
    -- Attention: quand l'émotion à reconnaître est " Surprise, Happy ou Sad ", la probabilité est au voisinage proche de 1, 
    Mais quand c'est " Disgusted ", le réseau a tendance à hésiter avec " Sad ":
    pour cette raison, la fonction renvoie "Disgusted" dès lors que la probabilité de " Disgusted " est supérieur à 0,2 
    et que celle de "Sad" est la plus élevée
    '''
    n,m = prediction.shape
    max_index=0
    max_prob=0
    for i in range(n):
        for j in range(m):
            if prediction[i][j]>max_prob:
                max_index=j
                max_prob=prediction[i][j]
    if prediction[0][0]>0.2 and max_index==2:
        return('Disgusted',max_prob)
    return(list(emotion_code.keys())[max_index],max_prob)

def predict_for_noob(img,cnn):
    ''' Argument: image à tester et réseau de neurone:
    3 étapes:
    1- Met l'image aux dimensions voulues
    2- Fait une prédiction
    3- Décode la prédiction et renvoie l'émotion reconnue ainsi que la probabilité associée'''
    processed_faces_list = process_for_model(img)
    n = len(processed_faces_list)
    if n == 0:
        return('Pas de visage détecté')
    elif n == 1:
        proc_img = processed_faces_list[0]
        prediction = predict(proc_img,cnn)
        return decode_predictions(prediction)
    else:
        emotion = {}
        emotion['Disgusted'] = 0
        emotion['Happy'] = 0
        emotion['Sad'] = 0
        emotion['Surprised'] = 0
        for face in processed_faces_list:
            prediction = predict(face,cnn)
            emotion[decode_predictions(prediction)[0]] += 1
        return emotion
