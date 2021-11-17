# M1-DataMining
#### Frank Doronzo, Romain Dudoit, Nathan Coustance

## Programmes de déploiement
### Système de classement => Classement/(arbre.py & dtree.sav)
Dans la console : python arbre.py  
Vous devrez ensuite rentrer le chemin absolu de votre base puis le programme se chargera
de générer un fichier predictions.txt qui contient la colonne y_pred soit la prédiction pour V200.

### Système de scoring => Scoring/(scoring.py & LDA.sav)
Ouvrir le fichier scoring.py dans Spyder puis exécuter le script (F5).  
Vous devrez ensuite rentrer le chemin absolu de votre base dans la console de Spyder puis
le programme se chargera de générer un fichier Score.csv qui contient le score
d'appartenance à la classe cible 'm16'.

## Autres fichiers
### Rapport.pdf
Fichier pdf de notre rapport pour ce projet

### featureselection.py
Il s'agit du fichier dans lequel se trouve le code que nous avons utilisé afin d'effectuer notre
sélection de variables.

### Classement/dtree_model_eval.ipynb
Il s'agit d'un notebook Jupyter dans lequel on trouve du code ainsi que des commentaires
sur la modélisation et l'évaluation de notre méthode de classement.
