import cv2
import os
import numpy as np

#Le decimos la ruta donde se encuentran las carpetas de los modelos ya capturados
dataPath = 'C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/Rostros'  # SE CAMBIA LA RUTA

#retorna una lista con los nombres de las carpetas creadas
peoplelist = os.listdir(dataPath)
#print("Lista personas", peoplelist)

#Crear un array para poder indentificar mediante posiciones las diferntes personas
labels = []
facesDate = []
label = 0

#1 especificamos la ruta del directorio donde se van a leer las imagenes
for nameDir in peoplelist:
    personPath = dataPath + "/" + nameDir
    print("Leyendo Imagenes")

#2 leemos las imagenes de la carpeta
    for fileName in os.listdir(personPath):
        print("Rostros: ",  nameDir + "/" + fileName)
        #Asignar las posiciones de cada imagen
        labels.append(label)
        #las pasamos a escala de grises
        facesDate.append(cv2.imread(personPath+ "/" + fileName, 0))
        image = cv2.imread(personPath + "/" + fileName, 0)
        #motrar las imagenes capturadas
        cv2.imshow("image", image)
        cv2.waitKey(10)
    #aumentamos el contador de las etiquetas
    label += 1
#print("labels = ", labels)

#utilizamos el Metodo
faces = cv2.face.EigenFaceRecognizer_create()
print("Enetreando modelo...")
#Entrenamiento del modelo
faces.train(facesDate, np.array(labels ))
#Almacenando modelo
faces.write("modeloEntrenado.xml")
print("Modelo almacenado....")