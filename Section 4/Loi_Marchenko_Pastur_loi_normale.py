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
    Facteur = 1
if indice_simulation == 2:
    N = 100
    Facteur = 2
if indice_simulation == 3:
    N = 250
    Facteur = 5
if indice_simulation == 4:
    N = 500
    Facteur = 10
if indice_simulation == 5:
    N = 1000
    Facteur = 20
if indice_simulation == 6:
    N = 10000
    Facteur = 200


nombre_lignes = 50
nombre_colonnes = 150


#Création de la matrice aléatoire
H = np.random.normal(0, 1, size=(Facteur*nombre_lignes, Facteur*nombre_colonnes))


gamma = (Facteur*nombre_lignes)/(Facteur*nombre_colonnes)
gamma_plus = (1 + np.sqrt(gamma))**2
gamma_moins = (1 - np.sqrt(gamma))**2

#Création d'un paramètre pour tracer la fonction de densité théorique attendue
t = np.linspace(gamma_moins, gamma_plus, 1000)

W = (1/(Facteur*nombre_colonnes))*(H @ np.transpose(H))

#Vecteur de valeurs propres
valeurs_propres = np.linalg.eig(W)[0]
print("Temps d'exécution :",(time() - temps_initial))

#Tracage de l'histogramme des valeurs propres de la matrice aléatoire
plt.hist(valeurs_propres, density=True, bins=50, color='blue', edgecolor='black')
plt.plot(t, (np.sqrt((gamma_plus-t)*(t-gamma_moins)))/(2*np.pi*gamma*t), color='red')
plt.show()

