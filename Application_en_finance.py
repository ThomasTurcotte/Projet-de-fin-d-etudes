import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


# Recherche des indices des sociétés du S&P500
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
liste_tickers = tickers.Symbol.to_list()


# On enlève les indices dont les données ne sont pas récupérables
liste_tickers.remove('BRK.B')
liste_tickers.remove('BF.B')


# Téléchargement des données et création d'une matrice
donnees = yf.download(liste_tickers,'2020-01-01','2025-01-01', auto_adjust=True)['Close']
Matrice_donnees = np.transpose(donnees.to_numpy())


# On enlève les sociétés dont certaines des valeurs de fermeture ne sont pas disponible sur la période de temps choisie
Liste_lignes = []
for j in range(Matrice_donnees.shape[1]):
    for i in range(Matrice_donnees.shape[0]):
        if np.isnan(Matrice_donnees[i, j]) == True:
            if i not in Liste_lignes:
                Liste_lignes.append(i)

compteur = 0
for indice in Liste_lignes:
    Matrice_donnees = np.delete(Matrice_donnees, indice - compteur, 0)
    compteur = compteur + 1


# Calcul des variations entre chacune des journées ou la bourse est ouverte
for i in range(Matrice_donnees.shape[1] - 1):
    Matrice_donnees[:, i] = Matrice_donnees[:, i + 1] - Matrice_donnees[:, i]

Matrice_donnees_finale = np.delete(Matrice_donnees, Matrice_donnees.shape[1] - 1, 1)
print('Dimensions de la matrice de données finale = ', Matrice_donnees_finale.shape)


# Standardisation de la matrice de données et calcul de la matrice de covariance
ecart_type = np.std(Matrice_donnees_finale)
moyenne = np.mean(Matrice_donnees_finale)
Matrice_donnees_finale = (Matrice_donnees_finale - Matrice_donnees_finale.mean(1, keepdims=True))/Matrice_donnees_finale.std(1, keepdims=True)
Matrice_covariance = (1/Matrice_donnees.shape[1])*(Matrice_donnees_finale @ np.transpose(Matrice_donnees_finale))


# Paramètre gamma pour Marchenko-Pastur
gamma = Matrice_donnees_finale.shape[0]/Matrice_donnees_finale.shape[1]
gamma_plus = (1 + np.sqrt(gamma))**2
gamma_moins = (1 - np.sqrt(gamma))**2
t = np.linspace(gamma_moins, gamma_plus, 1000)


# Calcul et traitement des valeurs propres
Valeurs_propres = np.linalg.eig(Matrice_covariance)[0]
Liste = []
for vp in Valeurs_propres:
    if (vp < gamma_plus) and (vp > gamma_moins):
        Liste.append(vp)


print('Valeurs propres : \n', np.sort(Valeurs_propres))
print('Nombre de valeurs propres total : ', Valeurs_propres.shape[0])
print('Nombre de valeurs propres admissibles : ', len(Liste))
plt.hist(Liste, density=True, bins=30,  color='blue', edgecolor='black')
plt.plot(t, (np.sqrt((gamma_plus-t)*(t-gamma_moins)))/(2*np.pi*gamma*t), color='red')
plt.show()