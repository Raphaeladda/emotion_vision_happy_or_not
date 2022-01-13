## Ce programme prend une photo avec la webcam et enregistre son chemin dans une variable "a",
## sous forme de chaîne de caractères. La webcam s'ouvre. Pour prendre et sauvegarder une photo, appuyer sur "s".
## Pour fermer la webcam, appuyer sur "q".

import cv2 
import pathlib
import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from utils_cv.face_detection import face_detect
from utils_cv.image_processing_operations import resize_image, changing_color_mode



def prise_photo():
    ''' Active la WebCame, prend une photo en appuyant sur la touche s du clavier et renvoie l'image capturée'''
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read() 
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s') or key == ord(' '): 
                picture = frame.copy()
                cv2.destroyAllWindows()
                
                webcam.release()
                #cv2.imshow("Captured Image", picture)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                break 
            
            elif key == ord('q'):
            ##("Turning off camera.")
                webcam.release()
            ##("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    
    picture1 = np.array(picture)   
    return(picture1)

