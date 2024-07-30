import cv2 as cv #OpenCV
import matplotlib.pyplot as plt
import numpy as np 
import os

def mostrar_imagem(img):
    # Abre a imagem sem deixar a cor negativa
    imagem_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)

def salvar_imagens(imagens, caminho):
    if not os.path.exists(caminho): #Cria o diretorio passado se não existir
        os.mkdir(caminho)

    index = len(os.listdir(caminho)) # pega a quantidade de imgs na pasta para nomear

    for imagem in imagens: # Percorre a lista de images
        cv.imwrite(f'{caminho}/face{index}.jpg', imagem)
        index += 1

## PRE PROCESSAMENTO

imagem_pessoas = cv.imread('pessoas.jpg')

imagem_cinza = cv.cvtColor(imagem_pessoas, cv.COLOR_BGR2GRAY)

features_haar = "haarcascade_frontalface_alt2.xml"
caminho = f'{cv.data.haarcascades}/{features_haar}'

classificador = cv.CascadeClassifier(caminho)

faces = classificador.detectMultiScale(imagem_cinza)

imagem_copia = np.array(imagem_pessoas)

for x,y,w,h in faces:
    cv.rectangle(imagem_copia, (x,y), (x+w, y+h), (0,255,0), 2)

### Cortando as faces

imagens_cortadas = list()

for x, y, w, h in faces: 
    face = imagem_pessoas[y:y+h, x:x+w]  # Coordenada Y de Y até H e X de X até W
    face = cv.resize(face, (160,160)) # Padroniza tamanho da imagem das faces
    imagens_cortadas.append(face) # Coloca na lista

print(len(imagens_cortadas))

for indice, img in enumerate(imagens_cortadas):
    # Printa o tamanho de cada imagem pra confirmar a padronizacao
    print(img.shape)

    ## MOSTRA AS IMAGENS
    cv.imshow(f'Imagem {indice} cortada', img)
    mostrar_imagem(imagens_cortadas[indice])

salvar_imagens(imagens_cortadas, './Faces')
