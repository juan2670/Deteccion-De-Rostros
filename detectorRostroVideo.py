import cv2
import os
import imutils

# Configuración
personName = 'juan'  # Cambiar según la persona
dataPath = 'C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/Rostros'
haarcascade_path = 'C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/haarcascade_frontalface_default.xml'
personPath = os.path.join(dataPath, personName)

# Crear carpeta si no existe
if not os.path.exists(personPath):
    os.makedirs(personPath)
    print(f'Carpeta creada: {personPath}')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(haarcascade_path)

count, positive_count, negative_count = 0, 0, 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            negative_count -= 1
            cv2.putText(frame, f"No reconocido: {negative_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
                count += 1
                positive_count += 1
                cv2.putText(frame, f"Verificado: {positive_count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        # Salir con Escape o al capturar 10 imágenes
        k = cv2.waitKey(1)
        if k == 27 or count >= 200:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"Proceso terminado. Imágenes capturadas: {count}")