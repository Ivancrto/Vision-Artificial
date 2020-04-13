# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:57:40 2020

@author: Ivanxrto
"""
from os import listdir
import numpy as np
import cv2
import matplotlib.pyplot as plt


imagenes_train = []
informacion_train = []

#1. En primer lugar hará falta programar un bucle que cargue las imágenes de training. 
def cargarImagen(dateImage):
    for img in listdir("./train"):
         #1.1 La carga debería realizarse en niveles de gris. Esto se consigue poniendo a 0 el segundo
             #parámetro del comando cv2.imread. 
         img_grey = cv2.imread("./train/" + img, 0)
         dateImage.append(img_grey)
         
#2. Utilizar la clase cv2.ORB_create para obtener los keypoints y los descriptores de cada imagen. 
detector = cv2.ORB_create(100, 1.1, 1) #keypoints	scaleFactor    nlevels(pirámide) 
   
         
def main():
    print("Clase principal")
    cargarImagen(imagenes_train)
    
    
if __name__ == "__main__":
    main()
    