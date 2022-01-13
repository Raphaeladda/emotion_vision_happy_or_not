import cv2
import numpy as np
import os
# Iteration 1

def load(filename):
    '''Telecharge l'image sous forte de vecteur numpy à partir de son chemin'''
    return(cv2.imread(filename,1))

def resize_image(img,newdim):
    '''Redimensionne l'image'''
    resized_img = cv2.resize(img,(newdim[0], newdim[1]), interpolation = cv2.INTER_CUBIC)
    return(resized_img)

# Iteration 2
def rotate_image(img,degree):
    '''Effectue une rotation de l'image'''
    rows,cols = img.shape[0],img.shape[1]
    Matrix = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    dst = cv2.warpAffine(img,Matrix,(cols,rows))
    return(dst)
#Iteration3

def smoothing_image(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    return(blur)

# Iteration 4
def draw_rectangle(img,tlcorner,brcorner,color,line_thickness):
    '''Dessine un rectangle à partir des points données en argument: tl et br corner'''
    img = cv2.rectangle(img,tlcorner,brcorner,color,line_thickness)
    return(img)
    
    
def changing_color_mode(img,flag):
    '''Changle la couleur de l'image'''
    return(cv2.cvtColor(img,flag))

def grad(img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    return(laplacian)
