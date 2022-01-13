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



emotion_code={}
emotion_code['Disgusted']=0
emotion_code['Happy']=1
emotion_code['Sad']=2
emotion_code['Surprise']=3

def create_emotion_vectors(emotion):
    ''' Cette fonction prend en argument une émotion parmi ces quatres émotions:
    "Disgusted", "Happy", "Sad" ,"Surprise" et renvoie deux listes:
    - la liste x, qui comprend les vecteurs des visages des images du dossier de l'émotion entrée en argument
    - la liste y, qui contient les labels correspondants
    les labels de chaque émotions sont définis dans le dictionaire  emotion_code en début de code

    exemple: create_emotion_vectors('Happy') renvoie (x,y) avec:
    x: une liste de vecteurs de dimensions (224,224,3) qui sont les visage des photos du dossier Happy
    y: une liste d'entiers qui sont les labels correspondant: ici le label correspondant à Happy est 1 donc y=[1,1,1,1,1,1]
    
    la fonction dimensionne les vecteurs de x afin qu'ils soient prêts à être entrés dans le réseau de neurone pour l'entrainer'''
    x=[]
    y=[]
    liste_files=os.listdir('dataset/'+emotion)
    if '.DS_Store' in liste_files:
        liste_files.remove('.DS_Store')
    for picture in liste_files:
        filename=os.getcwd()+'/dataset/'+ emotion +'/' + picture  
        img=load(filename)
        img_list=face_detect(img)
        if len(img_list)!=0:
            img=img_list[0]
            img=resize_image(img,(224,224))
            x.append(img)
            y.append(emotion_code[emotion])
    return(x,y)
    
if __name__ == "__main__":
    ''' Creation de fichier .npy qui contiennent les vecteurs des images de la base de données afin de ne pas devoir 
    refaire à chaque fois les opérations nécéssaires pour que les données soient opérationnels pour le réseau de neurone'''
     
    emotions_list= os.listdir('dataset')
    if '.DS_Store' in emotions_list:
        emotions_list.remove('.DS_Store')
        
    for emotion in emotions_list:
        x,y=create_emotion_vectors(emotion)
        np.save(os.getcwd()+'/emotion_recognition/x_'+str(emotion),np.array(x))
        np.save(os.getcwd()+'/emotion_recognition/y_'+str(emotion),np.array(y))
