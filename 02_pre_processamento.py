import cv2 as cv #OpenCV
import matplotlib.pyplot as plt
import numpy as np 

def mostrar_imagem(img):
    # Abre a imagem sem deixar a cor negativa
    imagem_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)

## PRE PROCESSAMENTO

# No pré-processamento sera necessario converter a imagem para cinza

imagem_pessoas = cv.imread('pessoas.jpg')

imagem_cinza = cv.cvtColor(imagem_pessoas, cv.COLOR_BGR2GRAY)
cv.imshow('Pessoas', imagem_cinza)
mostrar_imagem(imagem_cinza)

# Consultando a documentação do OPEN CV existem features de Haar
# Haar -> Algoritimo que é utilizado para detecção de faces
features_haar = "haarcascade_frontalface_alt2.xml"
caminho = f'{cv.data.haarcascades}/{features_haar}'

# Instanciando o modelo de classificação passando as features como param
classificador = cv.CascadeClassifier(caminho)

# Com o modelo já treinado, usando o metodo detect multi scale
# Retorna um array de coordenadas que se referem a posição 
#           de onde as faces foram detectadas
faces = classificador.detectMultiScale(imagem_cinza)

imagem_copia = np.array(imagem_pessoas)

# x -> ponto inicial da largura da face encontrada
# y -> ponto inicial da altura da face encontrada
# w -> ponto final da largura da face encontrada
# h -> ponto final da altura da face encontrada
for x,y,w,h in faces:
    cv.rectangle(imagem_copia, (x,y), (x+w, y+h), (0,255,0), 2)

cv.imshow('Teste', imagem_copia)
mostrar_imagem(imagem_copia)
cv.imwrite('imagem_com_faces.jpg', imagem_copia)