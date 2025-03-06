import numpy as np
import matplotlib.pyplot as plt
from time import time
temps_initial = time()

#Taile de la matrice
N = 2000
#Choix de la matrice une matrice avec des entrées suivant une loi normale centrée réduite
# ou une matrice dont les entrées sont -1 ou 1, les deux avec une probababilité 1/2
Matrice = np.random.normal(0, 1/np.sqrt(N), size=(N,N))
#Matrice = np.random.choice([-1/np.sqrt(N),1/np.sqrt(N)],  size=(N, N))

#Calcul des valeurs prorpes de la matrice
valeurs_propres = np.linalg.eig(Matrice)[0]

#Séparation des valeurs propres en parties réelle et imaginaire pour faire la graphique
Re = valeurs_propres.real
Im = valeurs_propres.imag

#Paramètre qui va servir à tracer le bord du disque unité
t = np.linspace(0, 2*np.pi, 1000)

#Affichage du temps de calcul
print("Temps d'éxécution : ", time()-temps_initial)

#Création du graphique
plt.figure(figsize=(5,5))
plt.scatter(Re, Im, s=5, color='blue')
plt.plot(np.cos(t), np.sin(t), color='red', label=f'Courbe $\\vert z \\vert=1$')
plt.xlim(-1.25,1.25)
plt.ylim(-1.25, 1.25)
plt.xlabel('Réel')
plt.ylabel('Imaginaire')
plt.title(f"Distribution des valeurs propres d'une matrice de \n l'ensemble de Ginibre de dimension {N}$\\times${N}.")
plt.legend(loc='upper right')
plt.show()