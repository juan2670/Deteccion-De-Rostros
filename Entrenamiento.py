import cv2
import numpy as np
import os

# Ruta de las imágenes de entrenamiento
dataPath = 'C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/Rostros'

# Guardamos en una lista los nombres de cada una de las carpetas
imagePaths = os.listdir(dataPath)

# Crear el reconocedor
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Listas para almacenar las imágenes y etiquetas
facesData = []
labels = []

# Recorrer las imágenes para leerlas y etiquetarlas
for label, nameDir in enumerate(imagePaths):
    personPath = os.path.join(dataPath, nameDir)
    for fileName in os.listdir(personPath):
        if fileName.endswith(".jpg") or fileName.endswith(".png"):
            imgPath = os.path.join(personPath, fileName)
            gray = cv2.imread(imgPath, 0)
            facesData.append(gray)
            labels.append(label)

# Entrenar el modelo
face_recognizer.train(facesData, np.array(labels))

# Especifica la ruta completa donde se guardará el modelo dentro de la carpeta 'Rostros'
modelPath = os.path.join(dataPath, 'modeloLBPHFace.xml')

# Guardar el modelo entrenado en la carpeta 'Rostros'
face_recognizer.write(modelPath)

print("Modelo entrenado y guardado en:", modelPath)

