import cv2
import os
import imutils

#Creamos la carpeta donde se van a almacenar los rostros
personName = 'pedro'  # SE CAMBIA EL NOMBRE DE LA CARPETA DE LA PERSONA
dataPath = 'C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/Rostros'  # SE CAMBIA LA RUTA
personPath = os.path.join(dataPath, personName)


#En el caso de que no exista ninguna carpeta la vamos a crear la carpeta
if not os.path.exists(personPath):
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier('C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/haarcascade_frontalface_default.xml')
count = 0
positive_count = 0
negative_count = 0

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
        cv2.putText(frame, f"No reconocido: {negative_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join( personPath +'/rostro_{}.jpg'.format(count)), rostro)  # SE CAMBIA LA RUTA
            count += 1
            positive_count += 1
            cv2.putText(frame, f"Verificado: {positive_count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

#Le especificamos la cantidad de fotos que debe capturar
    k = cv2.waitKey(1)
    if k == 27 or count >= 10:
        break

cap.release()
cv2.destroyAllWindows()
