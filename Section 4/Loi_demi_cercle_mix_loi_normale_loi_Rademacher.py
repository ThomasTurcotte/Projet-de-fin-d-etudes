import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time

temps_initial = time()
#Taille de la amtrice
N = 10000

#Création de la matrice de Wigner
liste_indice = [0, 1]
B = np.random.choice(liste_indice, (N, N))
B = np.triu(B)
H = np.zeros((N, N))

for i in range(N):
    for j in range(i, N):
        if B[i,j] == 0:
            H[i,j] = np.random.normal(0, 1)
        if B[i,j] == 1:
            H[i,j] = np.random.choice([-1,1])

H_temporaire = np.transpose(deepcopy(H))

for i in range(N):
    H_temporaire[i,i] = 0

M = (1/np.sqrt(N))*(H + H_temporaire)

#Calcul des valeurs propres
valeurs_propres = np.linalg.eig(M)[0]
t = np.linspace(-2, 2, 1000)

print("Temps d'exécution : ", time() - temps_initial)

#Tracage de l'histogramme
plt.hist(valeurs_propres, bins=50, density=True, color='blue', edgecolor='black')
plt.plot(t, (1/(2*np.pi))*np.sqrt(4-t**2), color='red')
plt.show()
