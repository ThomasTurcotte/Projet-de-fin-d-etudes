from scipy.stats import special_ortho_group
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

#Fixation de l'échantillon pour pouvoir reproduire les résultats
np.random.seed(4)

#Taille de la matrice
N = 10

#Définition de la matrice orthogonale et sa transposée
Matrice_orthogonale = special_ortho_group.rvs(dim = N)
Mat_transition = deepcopy(Matrice_orthogonale)
Matrice_orthogonale_transp = np.transpose(Mat_transition)

#Listes vide qui vont contenir les valeurs prises par l'entrée 1,1 des deux matrices
liste_vide_mat_originale = []
liste_vide_mat_rotation = []

#Boucle pour aller chercher à plusieurs reprises les valeurs prises par l'entrée 1,1 des 2 matrices
for i in range(1000000):
    H = np.random.normal(0, 1/np.sqrt(2*N), size=(N, N))
    H_copie = deepcopy(H)
    H_transp = np.transpose(H_copie)
    M = H + H_transp
    Rot = Matrice_orthogonale @ M @ Matrice_orthogonale_transp
    liste_vide_mat_originale.append(M[0,0])
    liste_vide_mat_rotation.append(Rot[0,0])

#Création d'un paramètre pour tracer la fonction de densité théorique attendue
t = np.linspace(-2, 2, 1000)

#Histogramme pour faire le graphique
plt.hist(liste_vide_mat_originale, density=True, bins=50, color='blue', alpha=0.5, label='Matrice originale')
plt.hist(liste_vide_mat_rotation, density=True, bins=50, color='red', alpha=0.5, label='Matrice transformée')
plt.plot(t, 1/(np.sqrt(2/N)*np.sqrt(2*np.pi))*np.exp(-(1/(2*(2/N))*(t**2))), color='green', label=f'$f(t)=\\frac{{1}}{{\\sqrt{{2/{N}}}\\sqrt{{2\\pi}}}}e^{{-\\frac{{1}}{{2(2/{N})}}t^2}}$')
plt.legend(loc='upper right')
plt.title(f"Répartition des valeurs prises par l'entrée 1,1 \n des matrices de dimensions {N} $\\times$ {N}")
plt.xlabel('$t$')
plt.show()