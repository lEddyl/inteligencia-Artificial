import cv2
import numpy as np
from keras.models import load_model
import numpy as np

# Carga el modelo
model = load_model('C:\Custom-Object-Detection-Using-Python-OpenCV--Training-Database-using-TeachableMachine-main\Custom Object Detection\keras_Model.h5')

# CÁMARA puede ser 0 o 1 según la cámara predeterminada de su computadora.
camera = cv2.VideoCapture(0)

# Toma las etiquetas del archivo etiquetas.txt. 
labels = open('C:\Custom-Object-Detection-Using-Python-OpenCV--Training-Database-using-TeachableMachine-main\Custom Object Detection\labels.txt', 'r').readlines()

while True:
    # Toma las imagenes de la cámara web.
    ret, image = camera.read()
    # Cambia el tamaño de la imagen sin procesar a (224 alto, 224 ancho) píxeles.
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # Mostrar la imagen en una ventana.
    
    # Convierte la imagen en una matriz numpy y cámbiele la forma de entrada del modelo.
    image = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normaliza la matriz de imágenes
    image = (image / 127.5) - 1
    # Hace que el modelo prediga cuál es la imagen actual. "Model.predict"
    # Devuelve una serie de porcentajes. Ejemplo: [0.2,0.8] significa que es 20% seguro
    # Es la primera etiqueta y 80% segura es la segunda etiqueta.
    probabilities = model.predict(image)

    print(list(probabilities[0]))
        

    # Imprime cuál es la etiqueta de probabilidad de mayor valor
    # print(np.argmax(probabilities))
    for predictions in (list(probabilities[0])):
        if predictions>=0.99:
            cv2.putText(img, labels[np.argmax(probabilities)],(5,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
            print(labels[np.argmax(probabilities)])
    cv2.imshow('Webcam Image', img)
    # Salirr
    keyboard_input = cv2.waitKey(1)
    # 27 es el ASCII de la tecla esc de tu teclado.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()