import cv2 as cv #OpenCV
import matplotlib.pyplot as plt
import numpy as np 

### FUNC
def mostrar_imagem(img):
    # Abre a imagem sem deixar a cor RGB
    imagem_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)

### INICIANDO

imagem_pessoas_mascara = cv.imread('imagem_pessoas_mascara.png')
imagem_cinza = cv.cvtColor(imagem_pessoas_mascara, cv.COLOR_BGR2GRAY)

print(type(imagem_pessoas_mascara))

#cv.imshow('Pessoas', imagem_pessoas_mascara)

features_haar = "haarcascade_frontalface_alt2.xml"
caminho = f'{cv.data.haarcascades}/{features_haar}'

classificador = cv.CascadeClassifier(caminho)

faces = classificador.detectMultiScale(imagem_cinza)

imagem_copia = np.array(imagem_pessoas_mascara)

for x,y,w,h in faces:
    cv.rectangle(imagem_copia, (x,y), (x+w, y+h), (0,0,255), 2)

cv.imshow('Teste', imagem_copia)
mostrar_imagem(imagem_copia)

mostrar_imagem(imagem_pessoas_mascara)

cv.imwrite('./imagens_faces/imagem_reconhece_mascara.jpg', imagem_copia)

