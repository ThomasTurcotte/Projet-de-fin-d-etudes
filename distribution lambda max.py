import numpy as np
import matplotlib.pyplot as plt
from time import time

temps_initial = time()

#Fixation de l'échantillon pour pouvoir reproduire les résultats
np.random.seed(4)

#Taille de la matrice
N = 100

#Nombre de répétition de l'expérience
repetition = 1000

#Liste vide qui contient les valeurs propres maximales de chacune des répétitions
valeurs_propres_maximales = []

#Boucle qui permet d'aller chercher la valeur propre maximale pour chacun des essais
for i in range(repetition):
    H = np.random.normal(0, 1 / np.sqrt(2 * N), size=(N, N))
    M = H + np.transpose(H)
    valeurs_propres = np.linalg.eig(M)[0]
    valeurs_propres_maximales.append(np.max(valeurs_propres))
    print(i)

print("Temps d'exécution : ",time()-temps_initial)

#Histogramme des valeurs propres maximales
plt.hist(valeurs_propres_maximales, density=True, bins=30, color='blue', alpha=0.5, label='Matrice originale', edgecolor='black')
plt.title(f'Distribution des valeurs propres maximales \n de {repetition} matrices {N} $\\times$ {N}.')
plt.xlabel('Plus grande valeur propre')
plt.ylabel('Répétition')
plt.show()
