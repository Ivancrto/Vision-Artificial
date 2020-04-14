# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:57:40 2020

@author: Ivanxrto
"""
from os import listdir
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math 

#AQUI CREAMOS TODAS LAS ESTRUCTURAS QUE VAYAMOS A USAR
imagenes_train = []
informacion_train = []



#1. En primer lugar hará falta programar un bucle que cargue las imágenes de training. 
def cargar_imagen(dateImage):
    for img in listdir("./train"):
         #1.1 La carga debería realizarse en niveles de gris. Esto se consigue poniendo a 0 el segundo
             #parámetro del comando cv2.imread. 
        #En la posicion 0 guardo la imagen en niveles de gris y en la posicion 1 la imagen en color original 
         dateImage.append([cv2.imread("./train/" + img, 0), cv2.imread("./train/" + img)])
         
#2. Utilizar la clase cv2.ORB_create para obtener los keypoints y los descriptores de cada imagen. 
detector = cv2.ORB_create(100, 1.1, 1) #keypoints	scaleFactor    nlevels(pirámide) 
   
#CREAMOS un FlannBasedMatcher utilizando la distancia de Hamming
FLANN_INDEX_LSH = 6
#En la especificacion de OPENCV, pone los siguientes valores, a lo mejor los tenemos que modificar
#En las diapositivas del profesor table_number = 6,key_size = 3,multi_probe_level = 1
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=-1) # Maximum leafs to visit when searching for neighbours.
flann = cv2.FlannBasedMatcher(index_params,search_params)

def obtener_informacion(date_Information, img_train):
    for img in img_train:
        #Obtenmos los keyPoints y los descriptores de la imagen i
        kp , des =  detector.detectAndCompute(img[0], None)
        
        #CorX y CorY son las coordenadas de nuestra imagen img 
        corX = img[0].shape[0] / 2; #Ancho de la imagen partido entre dos
        corY = img[0].shape[1] / 2; #Largo de la imagen partido entre dos
        i1 = 0;
        for k in kp:
            #Creamos el arbol añadiendo los descriptores a FlannBasedMatcher
            flann.add(des)
            #https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html METODOS DEL OBJETO KEYPOINT
            distancia_puntos = math.sqrt((corX - k.pt[0])**2 + (corY - k.pt[1])**2)
            angulo_kp = k.angle
            #DISTANCIA ANGLE DESCRIPTOR SIZE(FALTA) 
            date_Information.append([img, distancia_puntos, angulo_kp, des[i1]])
            i1 +=1
       
    
         
def main():
    cargar_imagen(imagenes_train)
    obtener_informacion(informacion_train, imagenes_train)
    
if __name__ == "__main__":
    main()
    