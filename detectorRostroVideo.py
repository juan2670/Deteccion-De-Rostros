import cv2
import os
import imutils

# Creacion de la carpeta
personName = 'pedro'  # Cambiar según la persona
dataPath = 'C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/Rostros'
haarcascade_path = 'C:/Users/user/OneDrive/Escritorio/RECONOCIMIENTO/haarcascade_frontalface_default.xml'
personPath = os.path.join(dataPath, personName)


# Crear carpeta si no existe
if not os.path.exists(personPath):
    os.makedirs(personPath)
    print(f'Carpeta creada: {personPath}')

# Capturamos la camara del dispositivo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#utiliza el modelo calificado
faceClassif = cv2.CascadeClassifier(haarcascade_path)

#Valores cuando se detecta y cuando no se detecta
count, positive_count, negative_count = 0, 0, 0



try:
    # capturará frames de la cámara en tiempo real.

    #ret: Indica si se pudo leer el frame correctamente.
    #frame: Es la imagen capturada.
    while True:
        ret, frame = cap.read()
        # Si no se pudo capturar el frame (por ejemplo, si se desconecta la cámara), el bucle se rompe.
        if not ret:
            break
        #Redimensiona el frame capturado a un ancho de 640 píxeles para procesamiento uniforme.
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Se crea una copia del frame actual para manipularlo sin afectar la imagen original.
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    #Si no se detectan rostros
        if len(faces) == 0:
            #Incrementa el contador de intentos en los que no se detectó un rostro.
            negative_count -= 1

            #Muestra el texto "No reconocido" en la pantalla con el número de intentos fallidos.
            cv2.putText(frame, f"No reconocido: {negative_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        #Si se detectan rostros
        else:
            #Itera sobre cada rostro detectado.
            #(x, y): Coordenadas superiores izquierdas del rectángulo que encierra el rostro.
            #(w, h): Anchura y altura del rectángulo.
            for (x, y, w, h) in faces:
                # Dibuja un rectángulo verde alrededor del rostro detectado.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                #Extrae la región de interés (el rostro) de la copia del frame.
                rostro = auxFrame[y:y + h, x:x + w]

                #Redimensiona la imagen del rostro a 150x150 píxeles para almacenamiento o procesamiento uniforme.
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

                #Guarda la imagen del rostro detectado en una carpeta específica (personPath) con un nombre único (rostro_{count}.jpg).
                cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)

                #Incrementa el contador total de imágenes capturadas.
                count += 1
                positive_count += 1

                #Muestra el texto "Verificado" en la pantalla con el número de detecciones exitosas.
                cv2.putText(frame, f"Verificado: {positive_count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        #Mostrar el frame procesado:

        #Muestra el frame procesado (con rectángulos y texto) en una ventana llamada "frame".
        cv2.imshow('frame', frame)


        #Si se presiona Esc o se capturan al menos 200 imágenes, el bucle se rompe.
        k = cv2.waitKey(1)
        if k == 27 or count >= 200:
            break


finally:
    #Libera el dispositivo de captura de video
    cap.release()

    #Cierra todas las ventanas de OpenCV.
    cv2.destroyAllWindows()

#Muestra un mensaje indicando que el proceso terminó y cuántas imágenes se capturaron en total.
print(f"Proceso terminado. Imágenes capturadas: {count}")