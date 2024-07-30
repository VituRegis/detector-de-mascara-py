import cv2 as cv # OpenCV
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import os
import warnings

warnings.filterwarnings("ignore") # usado pra ignorar alguns avisos das bibliotecas

def carrega_dataframe():
    dados = {
        "ARQUIVO": [],
        "ROTULO": [],
        "ALVO": []
    }

    caminho_com_mascara = "./com-mascara"
    caminho_sem_mascara = "./sem-mascara"

    # Carrega arquivos do diretório com máscara
    for arquivo in os.listdir(caminho_com_mascara):
        if arquivo.endswith(".jpg") or arquivo.endswith(".png"):  # Ajuste para o tipo de arquivo de imagem que você tem
            dados["ARQUIVO"].append(f"{caminho_com_mascara}{os.sep}{arquivo}")
            dados["ROTULO"].append("Com mascara")
            dados["ALVO"].append(1)

    # Carrega arquivos do diretório sem máscara
    for arquivo in os.listdir(caminho_sem_mascara):
        if arquivo.endswith(".jpg") or arquivo.endswith(".png"):  # Ajuste para o tipo de arquivo de imagem que você tem
            dados["ARQUIVO"].append(f"{caminho_sem_mascara}{os.sep}{arquivo}")
            dados["ROTULO"].append("Sem mascara")
            dados["ALVO"].append(0)

    dataframe = pd.DataFrame(dados)

    return dataframe

def ler_imagens(dados):
    arquivos = dados["ARQUIVO"]
    imagens = list()

    for arquivo in arquivos:
        img = cv.imread(arquivo)
        if img is not None:
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).flatten()
            imagens.append(img_gray)
        else:
            print(f"Erro ao carregar imagem: {arquivo}")
            imagens.append(None)  # Adiciona um valor None ou pode escolher remover a entrada problemática

    dados["IMAGEM"] = imagens

dados = carrega_dataframe()
dados.to_csv("./imagens-df.csv", index=False)  # Salva sem o índice

dados = pd.read_csv("./imagens-df.csv") 

ler_imagens(dados)

print(dados.head())

X = list(dados["IMAGEM"])
y = list(dados["ALVO"])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=23)

pca = PCA(n_components=30)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

parametros = {
    "n_neighbors": [2, 3, 5, 11, 19, 23, 29],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattam", "cosine", "l1", "l2"]
}

knn = GridSearchCV(KNeighborsClassifier(), parametros)
knn.fit(X_train, y_train)

predicao = knn.predict(X_test)

print(knn.score(X=X_test, y=y_test))
print(predicao)

verdadeiros_positivos, falsos_posistivos, falsos_negativos, verdadeiros_negativos = confusion_matrix(y_test, predicao).ravel()

print(verdadeiros_positivos, verdadeiros_negativos)
print(falsos_posistivos, falsos_negativos)

classificador = cv.CascadeClassifier(f"{cv.data.haarcascades}/haarcascade_frontalface_alt2.xml")

def processar_imagens(pca, classificador, imagem):
    img = cv.imread(imagem)
    imagem_cinza = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = classificador.detectMultiScale(img)
    vetores = list()
    cont = 0
    fig = plt.figure(figsize=(10,10))
    for x, y, w, h in faces:
        if cont < 9:  # Adiciona subplots somente se cont for menor que 9
            face_cortada = imagem_cinza[y:y+h, x:x+w]
            fig.add_subplot(3, 3, cont+1)
            plt.imshow(cv.cvtColor(face_cortada, cv.COLOR_BGR2RGB))
            cont += 1
        face_cortada = cv.resize(face_cortada, (160, 160))
        vetor = face_cortada.flatten()
        vetores.append(vetor)

    plt.show()
    return vetores

classes = {
    0: "Sem mascara",
    1: "Com mascara"
}

vetores = processar_imagens(pca, classificador, "./testar_04/pessoas_rua.jpg")
c = knn.predict(pca.transform(vetores))

print(*[classes[e] for e in c], sep=" - ")

vetores = processar_imagens(pca, classificador, "./testar_04/pessoas_rua2.jpg")
c = knn.predict(pca.transform(vetores))

print(*[classes[e] for e in c], sep=" - ")

