import cv2 as cv #OpenCV
import matplotlib.pyplot as plt
import numpy as np 

def mostrar_imagem(img):
    # Abre a imagem sem deixar a cor RGB
    imagem_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)

### INICIANDO

imagem_pessoas = cv.imread('pessoas.png')
imagem_cinza = cv.cvtColor(imagem_pessoas, cv.COLOR_BGR2GRAY)

print(type(imagem_pessoas))

features_haar = "haarcascade_frontalface_alt2.xml"
caminho = f'{cv.data.haarcascades}/{features_haar}'

classificador = cv.CascadeClassifier(caminho)

faces = classificador.detectMultiScale(imagem_cinza)

imagem_copia = np.array(imagem_pessoas)

for x,y,w,h in faces:
    cv.rectangle(imagem_copia, (x,y), (x+w, y+h), (0,0,255), 2)

cv.imshow('Teste', imagem_copia)
mostrar_imagem(imagem_copia)

mostrar_imagem(imagem_pessoas)

cv.imwrite('imagem_reconhece,mascara.jpg', imagem_copia)

imagens_cortadas = list()

for x, y, w, h in faces: 
    face = imagem_pessoas[y:y+h, x:x+w]
    face = cv.resize(face, (160,160))
    imagens_cortadas.append(face)

print(len(imagens_cortadas))

for img in imagens_cortadas:
    print(img.shape)