import cv2
from tkinter import *
import tkinter.messagebox
import tkinter.filedialog
import random
from PIL import Image, ImageFont, ImageDraw, ImageTk

import pathlib
import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from utils_cv.face_detection import face_detect
from utils_cv.image_processing_operations import resize_image, changing_color_mode
from Photo_booth.take_picture import prise_photo
from emotion_recognition.prediction import predict_for_noob

from keras.models import load_model


def Ouvrir(cnn, Canevas,img_dict, Mafenetre,Ligne):
    Canevas.delete(ALL) # on efface la zone graphique
    Ligne.delete(ALL)

    filename = tkinter.filedialog.askopenfilename(title="Ouvrir une image",filetypes=[('jpeg files','.jpg'),('jpeg files','.jpeg'),('all files','.*')])
    
    im = Image.open(filename) 
    photo_predict = np.array(im)
    x_max, y_max = photo_predict.shape[:2]

    if x_max > 600:
        alpha = 600 / x_max
        x_new = 600
        y_new = int(y_max*alpha)
        print(x_max,y_max)
        print(y_new,x_new)
        im = im.resize((y_new,x_new),Image.ANTIALIAS)
     
    photo = ImageTk.PhotoImage(im) 
    img_dict[filename] = photo  # référence
    print(img_dict)

    Canevas.create_image(0,0,anchor=NW,image=photo)
    Canevas.config(height=photo.height(),width=photo.width())

    
    Ligne.pack()
    Ligne.config(height=30,width=350)
    
    result = predict_for_noob(photo_predict,cnn)

    if result == 'Pas de visage détecté':
        Ligne.create_text(10,10,anchor=NW,text = 'Pas de visage détecté')

    if str(type(result)) == "<class 'dict'>":
        nb_disgusted = str(result['Disgusted'])
        nb_happy = str(result['Happy'])
        nb_sad = str(result['Sad'])
        nb_surprised = str(result['Surprised'])
        msg = "Dégoûtés: " + nb_disgusted + " | Heureux: " + nb_happy + " | Triste: " + nb_sad + " | Surpris: " + nb_surprised
        Ligne.create_text(10,10,anchor=NW,text = msg)

    else:
        A = result [0]

        if A == "Disgusted":
            msg = "Dégoûté avec le probabilité: " + str(result[1])
            Ligne.create_text(10,10,anchor=NW,text = msg)
        elif A == "Happy":
            msg = "Heureux avec le probabilité: " + str(result[1])
            Ligne.create_text(10,10,anchor=NW,text = msg)
        elif A == "Sad":
            msg = "Triste avec le probabilité: " + str(result[1])
            Ligne.create_text(10,10,anchor=NW,text = msg)
        elif A == "Surprised":
            msg = "Surpris avec le probabilité: " + str(result[1])
            Ligne.create_text(10,10,anchor=NW,text = msg)


    Mafenetre.title("Image "+str(photo.width())+" x "+str(photo.height()))


def Fermer(Canevas,Ligne,Mafenetre):
    Canevas.delete(ALL)
    Ligne.delete(ALL)
    Mafenetre.title("Image")

def Apropos():
    tkinter.messagebox.showinfo("A propos","Programme de reconnaissance visuelle\nCoding Weeks 2019")

def Liste_sentiments():
    tkinter.messagebox.showinfo("Liste des sentiments", "Dégoûté\nHeureux\nTriste\nSurpris")

def Webcam(cnn, Canevas, img_dict, Mafenetre,Ligne):
    ##Canevas.create_window(0,0,anchor=NW,window=)
    Canevas.delete(ALL)
    Ligne.delete(ALL)

    photo_web = prise_photo()
    x_max, y_max = photo_web.shape[:2]

    RGB_img = cv2.cvtColor(photo_web, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(RGB_img)
    if x_max > 600:
        alpha = 600 / x_max
        x_new = 600
        y_new = int(y_max*alpha)
        im = im.resize((y_new,x_new),Image.ANTIALIAS)
     
    photo = ImageTk.PhotoImage(im)
    img_dict[str(photo_web)] = photo  # référence

    Canevas.create_image(0,0,anchor=NW,image=photo)
    Canevas.config(height=photo.height(),width=photo.width())
    
    Ligne.pack()
    Ligne.config(height=30,width=350)
    
    result = predict_for_noob(photo_web,cnn)

    if result == 'Pas de visage détecté':
        Ligne.create_text(10,10,anchor=NW,text = 'Pas de visage détecté')

    if str(type(result)) == "<class 'dict'>":
        nb_disgusted = str(result['Disgusted'])
        nb_happy = str(result['Happy'])
        nb_sad = str(result['Sad'])
        nb_surprised = str(result['Surprised'])
        msg = "Dégoûtés: " + nb_disgusted + " | Heureux: " + nb_happy + " | Triste: " + nb_sad + " | Surpris: " + nb_surprised
        Ligne.create_text(10,10,anchor=NW,text = msg)
        
    else:
        A = result [0]

        if A == "Disgusted":
            msg = "Dégoûté avec le probabilité: " + str(result[1])
            Ligne.create_text(10,10,anchor=NW,text = msg)
        elif A == "Happy":
            msg = "Heureux avec le probabilité: " + str(result[1])
            Ligne.create_text(10,10,anchor=NW,text = msg)
        elif A == "Sad":
            msg = "Triste avec le probabilité: " + str(result[1])
            Ligne.create_text(10,10,anchor=NW,text = msg)
        elif A == "Surprised":
            msg = "Surpris avec le probabilité: " + str(result[1])
            Ligne.create_text(10,10,anchor=NW,text = msg)


def lancement():
    # Main window
    Mafenetre = Tk()
    Mafenetre.title("image")
    Mafenetre['bg']='#19D8BF'


    #Chargement du réseau de neuronnes
    
    # Création d'un widget Canvas
    Canevas = Canvas(Mafenetre)
    Canevas.pack(padx=40,pady=40)
    Ligne=Canvas(Mafenetre)

    # Utilisation d'un dictionnaire pour conserver une référence
    img_dict={}
    cnn = load_model(os.getcwd() + '/emotion_recognition/neural_model_emotion.h5')
    #definition des fonctions spécifiques
    def Ouvrir_bis():
        Ouvrir(cnn,Canevas,img_dict,Mafenetre,Ligne)

    def Webcam_bis():
        Webcam(cnn,Canevas,img_dict,Mafenetre,Ligne)

    def Fermer_bis():
        Fermer(Canevas,Ligne,Mafenetre)
    
    def fin():
        Mafenetre.destroy()
        return()

    # Création d'un widget Menu
    menubar = Menu(Mafenetre)

    menufichier = Menu(menubar,tearoff=0)

    menufichier.add_command(label="Fermer l'image",command=Fermer)
    menufichier.add_command(label="Quitter",command=Mafenetre.destroy)
    menubar.add_cascade(label="Fichier", menu=menufichier)

    BoutonGo = Button(Mafenetre, text ='Ouvrir', command = Ouvrir_bis,cursor='hand')
    BoutonGo.pack(side = LEFT, padx = 10, pady = 10)

    BoutonQuitter = Button(Mafenetre, text ='Fermer', command = Fermer_bis,cursor='hand')
    BoutonQuitter.pack(side = LEFT, padx = 10, pady = 10)

    BoutonWebcam = Button(Mafenetre, text ='Webcam', command = Webcam_bis,cursor='hand')
    BoutonWebcam.pack(side = LEFT, padx = 10, pady = 10)

    menuaide = Menu(menubar,tearoff=0)
    menuaide.add_command(label="A propos",command=Apropos)
    menubar.add_cascade(label="Aide", menu=menuaide)
    menuaide.add_command(label="Liste des sentiments",command=Liste_sentiments)


    # Affichage du menu
    Mafenetre.config(menu=menubar)

    
    Mafenetre.mainloop()
    Mafenetre.destroy()
lancement()