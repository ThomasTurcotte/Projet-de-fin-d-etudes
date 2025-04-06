from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from time import time
temps_initial = time()

#Choix de la taille de la matrice
indice_simulation = 1

#Fixation de l'échantillon pour pouvoir reproduire les résultats
np.random.seed(4)

#Taille de la matrice
if indice_simulation == 1:
    N = 50
if indice_simulation == 2:
    N = 100
if indice_simulation == 3:
    N = 250
if indice_simulation == 4:
    N = 500
if indice_simulation == 5:
    N = 1000
if indice_simulation == 6:
    N = 10000

#Écart type
sigma = 1

#Création d'un paramètre pour tracer la fonction de densité théorique attendue
t = np.linspace(-2*sigma, 2*sigma, 1000)

#Création de la matrice aléatoire
H = np.random.normal(0, sigma / np.sqrt(2*N), size=(N, N))
H_copie = deepcopy(H)
H_transp = np.transpose(H_copie)
M = H + H_transp

#Vecteur de valeurs propres
valeurs_propres = np.linalg.eig(M)[0]
print("Temps d'exécution :",(time() - temps_initial))

#Tracage de l'histogramme des valeurs propres de la matrice aléatoire
plt.hist(valeurs_propres, density=True, bins=50, color='blue', edgecolor='black')
plt.plot(t, np.sqrt(4*(sigma**2)-t**2)/(2*np.pi*sigma**2), color='red')
plt.show()

