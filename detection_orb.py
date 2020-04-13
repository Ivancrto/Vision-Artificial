# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:57:40 2020

@author: Ivanxrto
"""
from os import listdir
import numpy as np
import cv2
import matplotlib.pyplot as plt

#AQUI CREAMOS TODAS LAS ESTRUCTURAS QUE VAYAMOS A USAR
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
   
#CREAMOS un FlannBasedMatcher utilizando la distancia de Hamming
FLANN_INDEX_LSH = 6
#En la especificacion de OPENCV, pone los siguientes valores, a lo mejor los tenemos que modificar
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=-1) # Maximum leafs to visit when searching for neighbours.
flann = cv2.FlannBasedMatcher(index_params,search_params)

         
def main():
    print("Clase principal")
    cargarImagen(imagenes_train)
    
    
if __name__ == "__main__":
    main()
    