import numpy as np
import matplotlib.pyplot as plt
from time import time

temps_initial = time()

#Fixation de l'échantillon pour pouvoir reproduire les résultats
np.random.seed(4)

#Liste vide qui va contenir l'espacement des valeurs propres
espacement = []

for i in range(100000):
    H = np.random.normal(0, 1 / np.sqrt(2), (2, 2))
    M = H + np.transpose(H)
    valeurs_propres = np.linalg.eig(M)[0]
    espacement.append(np.max(valeurs_propres)-np.min(valeurs_propres))

t = np.linspace(0, 9, 1000)

print("Temps d'exécution : ", time() - temps_initial)

#Traçage de l'histogramme
plt.hist(espacement, density=True, bins=50, color='blue', edgecolor='black')
plt.plot(t, (t/4)*np.exp(-t**2/8), color='red', label=f"$p(s)=\\frac{{se^{{-\\frac{{s^2}}{{8}}}}}}{{4}}$")
plt.legend(loc='upper right')
plt.xlabel('s', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.title("Distribution de l'espacement entre les deux valeurs propres \nd'une matrice $2\\times 2$ de l'ensemble gaussien orthogonal")
plt.show()
