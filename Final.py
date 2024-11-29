import cv2
import os

# Ruta de los datos y del modelo entrenado
dataPath = 'C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/Rostros'  # Cambiar según tu ubicación
modelPath = 'C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/Rostros/modeloLBPHFace.xml'

# Validar rutas
if not os.path.exists(modelPath):
    print("Error: Modelo LBPH no encontrado.")
    exit()

if not os.path.exists(dataPath):
    print(f"Error: Ruta {dataPath} no encontrada.")
    exit()

# Cargar nombres de personas
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# Inicializar el reconocedor y cargar el modelo entrenado
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(modelPath)

# Configurar la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargar el clasificador Haarcascade
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bucle de reconocimiento
while True:
    ret, frame = cap.read()
    if ret == False:
        print("Error al acceder a la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    # Detectar rostros
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        # Mostrar resultados
        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 70:  # Ajustar umbral según resultados
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Presionar ESC para salir
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()