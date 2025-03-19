import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time

temps_initial = time()
#Choix de la taille de la matrice
indice_simulation = 2

#Fixation de l'échantillon pour pouvoir reproduire les résultats
np.random.seed(4)

#Taille de la matrice
if indice_simulation == 1:
    N = 100
if indice_simulation == 2:
    N = 2000
if indice_simulation == 3:
    N = 10000
if indice_simulation == 4:
    N = 20000

#Écart type et paramètre de la loi exponentielle
parametre = 2
sigma = np.sqrt(parametre)


#Création de la matrice aléatoire

H = np.random.poisson(parametre, size=(N,N))
H_triangulaire = np.triu(H)
H_copie = deepcopy(H_triangulaire)
H_temporaire = np.transpose(H_copie)

for i in range(N):
    H_temporaire[i, i] = 0

M = 1/(np.sqrt(N))*(H_triangulaire + H_temporaire)

#Création d'un paramètre pour tracer la fonction de densité théorique attendue
t = np.linspace(-2*sigma, 2*sigma, 1000)

#Création de l'histogramme des valeurs propres
valeurs_propres = np.linalg.eig(M)[0]
print('Vecteur de Valeurs propres : ', np.sort(valeurs_propres))
print("Temps d'exécution : ", time() - temps_initial)
plt.hist(valeurs_propres, density=True, bins=100, color='blue', edgecolor='black')
plt.plot(t, 1/(2*np.pi*sigma**2)*np.sqrt(4*sigma**2 - t**2), color='red')
plt.show()