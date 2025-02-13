from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

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

#Création d'un paramètre pour tracer la fonction de densité théorique attendue
t = np.linspace(-2*sigma, 2*sigma, 1000)

#Création de la matrice aléatoire
H = np.random.normal(0, sigma / np.sqrt(2 * N), size=(N, N))
H_copie = deepcopy(H)
H_transp = np.transpose(H_copie)
M = H + H_transp

#Vecteur de valeurs propres
valeurs_prores = np.linalg.eig(M)[0]

#Tracage de l'histogramme des valeurs propres de la matrice aléatoire
plt.hist(valeurs_prores, density=True, bins=50, edgecolor='black')
plt.plot(t, np.sqrt(4*(sigma**2)-t**2)/(2*np.pi*sigma**2), color='red', label=f'$f(t)=\\frac{{\\sqrt{{4({sigma})^2-t^2}}}}{{2\\pi ({sigma})^2}}$')
plt.legend()
plt.title(f'Histrogramme des valeurs propres de la matrice \n aléatoire de dimension {N} $\\times$ {N}')
plt.xlabel('$t$')
plt.xlim((-2.5*sigma), (2.5*sigma))
plt.show()