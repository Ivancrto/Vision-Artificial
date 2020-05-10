# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:50:58 2020

@author: Ivanxrto
"""
from os import listdir
import numpy as np
import cv2
import matplotlib.pyplot as plt

imagenes = []
#Crear un objeto de tipo cv2.CascadeClassifier.
frontal_cascade = cv2.CascadeClassifier('./haar/coches.xml')

def cargar_imagen(date_image):
    for img in listdir("./test"):
         #1.1 La carga debería realizarse en niveles de gris. Esto se consigue poniendo a 0 el segundo
             #parámetro del comando cv2.imread. 
         img_grey = cv2.imread("./test/" + img, 0)
         img_color = cv2.imread("./test/" + img)
         date_image.append([img_grey, img_color])

def monstrar_recuadro(date_image):
    indice = 1
    for i in date_image:
        #Utilizar el método cv2.detectMultiScale para obtener los rectángulos donde el sistema detecta coches
        name_img ='Coche_frontal' + str(indice) 
        imagenP = frontal_cascade.detectMultiScale (i[0], 1.1, 11) # 1parametro imagen
        #2 parametro la escala
        #3 parametro minNeighbours 
        for (x,y,w,h) in imagenP:
            cv2.rectangle(i[1], (x,y), (x+w,y+h), (100, 255, 200), 2)
        if imagenP is ():
            print("Frontal no encontrado en la imagen: " + name_img)
        cv2.imshow(name_img, i[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        indice += 1

def main():
    print("Clase principal")
    cargar_imagen(imagenes)
    monstrar_recuadro(imagenes)


if __name__ == "__main__":
    main()
