# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:59:13 2020

@author: Ivanxrto
"""

from os import listdir
import cv2



#Estructura para almacenar los videos
video = []
#Crear un objeto de tipo cv2.CascadeClassifier.
frontal_cascade = cv2.CascadeClassifier('./haar/coches.xml')

#Cargamos todos los videos de la carpeta, en nuestro array video
def cargar_videos(date_video):
    for v in listdir("./videos"):
         #VideoCapture nos ayuda a capturar el video
         video = cv2.VideoCapture("./videos/" + v)
         date_video.append(video)
         
def monstrar_recuadro_video(date_image):
    #Recorremos los dos videos,o todos los videos que tengamos en la carpeta
    for i in date_image:
        #Hacemos un bucle para que este leyendo continuamente      
        while True:
            #Con el metodo read obtenemos en res el resultado y en imagen la imagen capturada
            res, imagen = i.read()   
            imagenP = frontal_cascade.detectMultiScale(imagen, 1.09, 11)
            #Dibujamos el rectangul
            for (x, y, w, h) in imagenP:
                cv2.rectangle(imagen, (x, y), (x+w, y+h), (100, 255, 200), 2)               
            cv2.imshow("Video_Coche",imagen)
            tecla = cv2.waitKey(1)
            #Para cerrar el video o pasar al siguiente, pulsamos la tecla 27 (ESC)
            if tecla == 27:
                break
    i.release()
    cv2.destroyAllWindows()

            

def main():
    cargar_videos(video)
    monstrar_recuadro_video(video)
    
if __name__ == "__main__":
    main()