import sys
import os
sys.path.append(os.getcwd())
import cv2
from utils_cv.image_processing_operations import *

def face_detect(img): ##Detection du visage: retourne les visages d'une image
    ''' Prend une image en argument et renvoie la liste des visages détectés: chaque visage à sa propre image correspondante
        (sous forme de vecteur Numpy)'''
    faces_list = []
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(os.getcwd()+'/utils_cv/haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray_img=changing_color_mode(img,cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_img)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        img=draw_rectangle(img,(x, y),(x+w, y+h),(255, 0, 0),2) # Trace un rectangle à partir des 4 points detectés
        crop_img=img[y:y+h,x:x+w] # Coupe l'image en ne garde que le rectangle
        faces_list.append(crop_img)
    # Display the output
    return(faces_list)
