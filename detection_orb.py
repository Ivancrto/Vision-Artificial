from os import listdir
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# AQUI CREAMOS TODAS LAS ESTRUCTURAS QUE VAYAMOS A USAR
imagenes_train = []
imagenes_test = []


# 1. En primer lugar hará falta programar un bucle que cargue las imágenes de training.
def cargar_imagen(dateImage, path):
    for img in listdir(path):
        # 1.1 La carga debería realizarse en niveles de gris. Esto se consigue poniendo a 0 el segundo
        # parámetro del comando cv2.imread.
        # En la posicion 0 guardo la imagen en niveles de gris y en la posicion 1 la imagen en color original
        dateImage.append([cv2.imread(path + "/" + img, 0), cv2.imread(path + "/" + img), img])


# 2. Utilizar la clase cv2.ORB_create para obtener los keypoints y los descriptores de cada imagen.
detector = cv2.ORB_create(500, 1.3, 4)  # keypoints	scaleFactor    nlevels(pirámide)

# CREAMOS un FlannBasedMatcher utilizando la distancia de Hamming
FLANN_INDEX_LSH = 6
# En la especificacion de OPENCV, pone los siguientes valores, a lo mejor los tenemos que modificar
# En las diapositivas del profesor table_number = 6,key_size = 3,multi_probe_level = 1
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=10,  # 20
                    multi_probe_level=2)  # 2
search_params = dict(checks=-1)  # Maximum leafs to visit when searching for neighbours.
flann = cv2.FlannBasedMatcher(index_params, search_params)


def obtener_informacion(img_train):
    train_information = []
    for img in img_train:
        # Obtenmos los keyPoints y los descriptores de la imagen i
        kp, des = detector.detectAndCompute(img[0], None)
        distancia_puntos = {}
        # CorX y CorY son las coordenadas de nuestra imagen img
        corX = img[0].shape[0] / 2;  # Ancho de la imagen partido entre dos
        corY = img[0].shape[1] / 2;  # Largo de la imagen partido entre dos
        for k in kp:
            # Creamos el arbol añadiendo los descriptores a FlannBasedMatcher
            flann.add(des)
            # Distancia al centro de la imagen
            distancia_puntos[k] = [(corX - k.pt[0]), (corY - k.pt[1])]
        train_information.append([img, distancia_puntos, kp, des])
    return train_information


def procesar_imagen(img_test, informacion):
    for test in img_test:
        # Crear la matriz de votos
        votos = np.zeros((math.trunc(test[0].shape[0] / 10), math.trunc(test[0].shape[1] / 10), 1), np.uint8)
        # Detectar los keypoints y descriptores de las imagenes de test
        kpTest, desTest = detector.detectAndCompute(test[0], None)
        # Buscar los k=2 vectores mas cercanos
        for info in informacion:
            desTrain = info[3]
            kpTrain = info[2]
            good_matches = flann.knnMatch(desTest, desTrain, k=2)
            for m1, m2 in good_matches:
                # Los keypoints que coinciden
                kpTest1 = kpTest[m1.queryIdx]
                kpTest2 = kpTest[m2.queryIdx]
                kpTrain1 = kpTrain[m1.trainIdx]
                kpTrain2 = kpTrain[m2.trainIdx]

                # Angulos de la rotación de los descriptores en radianes
                rotTest1 = np.radians(kpTest1.angle)
                rotTest2 = np.radians(kpTest2.angle)
                rotTrain1 = np.radians(kpTrain1.angle)
                rotTrain2 = np.radians(kpTrain2.angle)

                # Tamaños de los descriptores
                scaleTrain1 = kpTrain1.size
                scaleTest1 = kpTest1.size
                scaleTrain2 = kpTrain2.size
                scaleTest2 = kpTest2.size

                # Las coordenadas de los descriptores del entrenamiento
                coordenadaXtrain1 = info[1][kpTrain1][0]
                coordenadaYtrain1 = info[1][kpTrain1][1]
                coordenadaXtrain2 = info[1][kpTrain2][0]
                coordenadaYtrain2 = info[1][kpTrain2][1]

                # Angulo de los descriptores del entrenamiento
                angleTrain1 = math.atan2(coordenadaYtrain1, coordenadaXtrain1)
                angleTrain2 = math.atan2(coordenadaYtrain2, coordenadaXtrain2)

                # Las coordenadas escaladas
                coordenadaXtrainEsc1 = kpTest1.pt[0] + coordenadaXtrain1 * scaleTest1/scaleTrain1
                coordenadaYtrainEsc1 = kpTest1.pt[1] + coordenadaYtrain1 * scaleTest1/scaleTrain1
                coordenadaXtrainEsc2 = kpTest2.pt[0] + coordenadaXtrain2 * scaleTest2/scaleTrain2
                coordenadaYtrainEsc2 = kpTest2.pt[1] + coordenadaYtrain2 * scaleTest2/scaleTrain2

                # Crear el vector nuevo escalado
                vector1 = np.array([[coordenadaXtrainEsc1], [coordenadaYtrainEsc1]])
                vector2 = np.array([[coordenadaXtrainEsc2], [coordenadaYtrainEsc2]])

                # Corregir el angulo segun la formula(sumar el angulo de la rotación del entrenamiento y restar el angulo de la rotación del test)
                newAngle1 = angleTrain1 + rotTest1 - rotTrain1
                newAngle2 = angleTrain2 + rotTest2 - rotTrain2

                # Crear matriz de rotacion segun los angulos nuevos corregidos
                R1 = np.array([[np.cos(newAngle1), -np.sin(newAngle1)], [np.sin(newAngle1), np.cos(newAngle1)]])
                R2 = np.array([[np.cos(newAngle2), -np.sin(newAngle2)], [np.sin(newAngle2), np.cos(newAngle2)]])

                # Multiplicar el matriz de rotacion por las coordenadas escaladas
                newVector1 = np.dot(R1, vector1)
                newVector2 = np.dot(R2, vector2)

                # Reducir la resolucion dividiendo entre 10 y encontrar las nuevas coordenadas del vector nuevo
                newX1 = math.trunc(newVector1[0][0]/10)
                newX2 = math.trunc(newVector2[0][0]/10)
                newY1 = math.trunc(newVector1[1][0]/10)
                newY2 = math.trunc(newVector2[1][0]/10)

                # Rellenar la matriz de votación
                if newX1 > 0 and newX1 < math.trunc(test[0].shape[0] / 10) and newY1 > 0 and newY1 < math.trunc(test[0].shape[1] / 10):
                    votos[newX1][newY1] = votos[newX1][newY1] + 1
                if newX2 > 0 and newX2 < math.trunc(test[0].shape[0] / 10) and newY2 > 0 and newY2 < math.trunc(test[0].shape[1] / 10):
                    votos[newX2][newY2] = votos[newX2][newY2] + 1

        # Buscar los maximos en la matriz de votación
        max_valores = buscar_maximos(votos)

        maximoX = max_valores[0] * 10
        maximoY = max_valores[1] * 10

        #Dibujar un cuadro en el centro encontrado del coche
        cv2.rectangle(test[1], (maximoX - math.trunc(test[1].shape[0]/4), maximoY + math.trunc(test[1].shape[1]/10)), (maximoX + math.trunc(test[1].shape[0]/4), maximoY - math.trunc(test[1].shape[1]/10)), (100, 255, 200), 2)
        cv2.imshow(test[2], test[1])
        cv2.waitKey()
        cv2.destroyAllWindows()

def buscar_maximos(votaciones):
    maxPosible = 0;
    maximo = -1;
    length1 = votaciones.shape[0]
    length2 = votaciones.shape[1]

    # Se busca los 4-8 maximos mas cercanos
    for i in range(0, length1):
        for j in range(0, length2):

            # Si estamos en la primera posicion de la matriz de votacion
            if (i == 0 and j == 0):
                maxPosible = votaciones[i][j] + votaciones[i + 1][j] + votaciones[i][j + 1] + votaciones[i + 1][j + 1]

            elif (j == 0):

                # Si estamos en la primera columna y en las filas interiores de la matriz de votacion
                if (0 < i < (length1 - 1)):
                    maxPosible = votaciones[i][j] + votaciones[i][j + 1] + votaciones[i - 1][j] + \
                                 votaciones[i + 1][j] + votaciones[i - 1][j + 1] + votaciones[i + 1][j + 1]

                # Si estamos en la primera columna y en la ultima fila de la matriz de votacion
                elif (i == (length1 - 1)):
                    maxPosible = votaciones[i][j] + votaciones[i - 1][j] + votaciones[i - 1][j + 1] + votaciones[i][j + 1]

            elif (i == 0):

                # Si estamos en la primera fila y en las columnas interiores de la matriz de votacion
                if (0 < j < (length2 - 1)):
                    maxPosible = votaciones[i][j] + votaciones[i][j - 1] + votaciones[i][j + 1] + \
                                 votaciones[i + 1][j - 1] + votaciones[i + 1][j] + votaciones[i + 1][j + 1]

                # Si estamos en la primera fila y en la ultima columna de la matriz de votacion
                elif (j == (length2 - 1)):
                    maxPosible = votaciones[i][j] + votaciones[i][j - 1] + votaciones[i + 1][j - 1] + votaciones[i + 1][j]

            # Si estamos en las filas interiores y en la ultima columna de la matriz de votacion
            elif (j == (length2 - 1)) and (0 < i < (length1 - 1)):
                maxPosible = votaciones[i][j] + votaciones[i - 1][j - 1] + votaciones[i - 1][j] + \
                             votaciones[i][j - 1] + votaciones[i + 1][j - 1] + votaciones[i + 1][j]

            # Si estamos en las columnas interiores y en la ultima fila de la matriz de votacion
            elif (i == (length1 - 1)) and (0 < j < (length2 - 1)):
                maxPosible = votaciones[i][j] + votaciones[i - 1][j - 1] + votaciones[i - 1][j] + \
                             votaciones[i - 1][j + 1] + votaciones[i][j - 1] + votaciones[i][j + 1]

            # Si estamos en la ultima fila y en la ultima columna de la matriz de votacion
            elif (i == (length1 - 1)) and (j == (length2 - 1)):
                maxPosible = votaciones[i][j] + votaciones[i - 1][j - 1] + votaciones[i - 1][j] + votaciones[i][j - 1]

            # Si estamos en las filas interiores y en las columnas interiores de la matriz de votacion
            else:
                maxPosible = votaciones[i][j] + votaciones[i - 1][j - 1] + votaciones[i - 1][j] + votaciones[i - 1][j + 1] + \
                             votaciones[i][j - 1] + votaciones[i][j + 1] + votaciones[i + 1][j - 1] + votaciones[i + 1][j] + votaciones[i + 1][j + 1]

            aux = np.max(votaciones);
            if maxPosible >= maximo:
                maximo = maxPosible
                maximoX = i
                maximoY = j

            elif votaciones[i][j] == aux and aux >= maximo:
                maximo = aux;
                maximoX = i
                maximoY = j

    return maximoX, maximoY


def main():
    cargar_imagen(imagenes_train, "./train")
    information = obtener_informacion(imagenes_train)
    cargar_imagen(imagenes_test, "./test")
    procesar_imagen(imagenes_test, information)


if __name__ == "__main__":
    main()


