Raphael ADDA
Eliott BIGIAOUI
Felix NADER
Lucas RAVIER
Frank NGUEUGA
Hugo LAULHERE


# Librairies Python requises
numpy ; keras ; sklearn ; sys ; os ; cv2 ; pathlib ; tkinter 

# Description du produit
Ce programme python est une application qui a été créé dans le contexte de la coding week 2019 de CentraleSupelec, qui a pour but de reconnaitre l'émotion d'un individu à partir d'une photographie choisie sur l'ordinateur ou prise par la webcam.
Nous avons utilisé un réseau de neurones, que nous avons construit à partir du modèle déjà préentrainé Resnet50, auquel on a rajouté quelques couches finales.
Notre set de données pour l'entraînement a été obtenu sur internet.
Pour des raisons de simplicité, notre logiciel peut reconnaître une émotion parmi la joie, la tristesse, la surprise et le dégoût. 
D'après les tests effectués sur notre programme, nous avons une précision de 80%. 

# Améliorations possibles 
Notre application est fonctionnelle en l'état.
Cependant, il serai possible d'en augmenter la précision en entraînant un réseau de neurones avec plus de données.
De plus, une amélioration future est d'élargir la reconnaissance à d'autres émotions, notamment la colère et la peur.

#Fonctionnement: 
Notre programme se lance en se plaçant dans le fichier "happy-or-not" sur le terminal, puis tapant la commande 'python3.'.
Une fenetre tkinter s'ouvre alors (elle charge quelques instants le temps que le réseau de neurone se télécharge).
On peut donc appuyer sur:
- le bouton Ouvrir, pour tester notre application sur une image pré-enregistré sur notre machine
- le bouton Webcam, pour tester notre application en prenant une photo avec la webcam de l'ordinateur

On peut répeter l'opération autant de fois quez souhaité en appuyant tout simplement à nouveau sur Webcam ou Ouvrir.
Pour arreter le programme , il suffit de fermet la fenetre tkinter.


# Python libraries required
numpy ; keras ; sklearn ; sys ; os ; cv2 ; pathlib ; tkinter 

# Product description
This python program is an application that as been created during the 2019 CentraleSupelec Coding Week, that aims to recognize the emotion of a person from a picture chosen on the computer or taken by the webcam.
We used a neural network, which we built from the already pre-trained Resnet50 model, to which we added some final layers.
Our training dataset was obtained from the internet.
For simplicity, our software can recognize one emotion among joy, sadness, surprise and disgust. 
According to the tests performed on our program, we have an accuracy of 80%. 

# Possible improvements 
Our application is functional.
However, it would be possible to increase the accuracy by training a neural network with more data.
Also, a future improvement is to extend the recognition to other emotions, including anger and fear.

#Function: 
Our program launches by going to the "happy-or-not" file on the terminal, then typing the command 'python3.'
A tkinter window opens (it loads for a few moments while the neural network downloads).
We can then press:
- the 'Ouvrir' button, to test our application on a pre-recorded image on our machine
- the 'Webcam' button, to test our application by taking a picture with the computer's webcam

You can repeat the operation as many times as you want by simply pressing again on Webcam or Open.
To stop the program, just close the tkinter window.

11/17/2019

Translated with www.DeepL.com/Translator (free version)