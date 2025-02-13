import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#Choix de la taille de la matrice
indice_simulation = 4

#Fixation de l'échantillon pour pouvoir reproduire les résultats
np.random.seed(4)

#Taille de la matrice
if indice_simulation == 1:
    N = 100
if indice_simulation == 2:
    N = 1000
if indice_simulation == 3:
    N = 10000
if indice_simulation == 4:
    N = 20000

#Écart type
sigma = 1

#Liste qui contient les valeurs possible pour les entrées de la matrice aléatoire
valeurs = [-1, 1]

#Création de la matrice aléatoire
H = np.random.choice(valeurs,  size=(N, N))
for i in range(N):
    for j in range(N):
        if j < i:
            H[i, j] = 0
H_copie = deepcopy(H)
H_temporaire = np.transpose(H_copie)
for i in range(N):
    H_temporaire[i, i] = 0
M = 1/(np.sqrt(N))*(H + H_temporaire)

#Création d'un paramètre pour tracer la fonction de densité théorique attendue
t = np.linspace(-2*sigma, 2*sigma, 1000)

#Création de l'histogramme des valeurs propres
valeurs_propres = np.linalg.eig(M)[0]
plt.hist(valeurs_propres, density=True, bins=50)
plt.plot(t, 1/(2*np.pi*sigma**2)*np.sqrt(4*sigma**2 - t**2))
plt.show()