import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
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


#Création de la matrice aléatoire
H = np.random.standard_cauchy(size=(N,N))
H_triangulaire = np.triu(H)
H_temporaire = np.transpose(deepcopy(H_triangulaire))

for i in range(N):
    H_temporaire[i, i] = 0

M = 1/(np.sqrt(N))*(H_triangulaire + H_temporaire)


#Création de l'histogramme des valeurs propres
valeurs_propres = np.linalg.eig(M)[0]
print("Plus petite valeur propre : ", np.min(valeurs_propres))
print("Plus grande valeur propre : ", np.max(valeurs_propres))
print("Temps d'exécution : ", time() - temps_initial)
plt.hist(valeurs_propres, density=True, bins=50, color='blue', edgecolor='black')
plt.show()