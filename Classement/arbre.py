import os
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


os.chdir(os.path.dirname(__file__))


file_path = input('Chemin absolu du fichier à charger : \n')

start_time = time.time()

# Variables explicatives pertinentes obtenus en amont après une sélection de variables
cols=['V14', 'V84', 'V96', 'V159', 'V160', 'V161', 'V162', 
      'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 
      'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V180', 'V181', 'V182', 'V183', 
      'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 
      'V192', 'V193', 'V194', 'V195', 'V197', 'V198', 'V199']

print("Chargement du fichier...")
#data = pd.read_csv(path,delimiter='\t',usecols=cols,nrows=4898424)

# recodage des variables qualitatives
data=pd.get_dummies(pd.read_csv(file_path,delimiter='\t',usecols=cols,nrows=4898424), columns=['V160','V162'],drop_first=True)
data[["V161"]]=data[["V161"]].apply(LabelEncoder().fit_transform) # transformation des variables qualitatives

print("Importation du modele")
f=open("dtree.sav","rb")
dtree=pickle.load(f)
f.close()

print("Calcul des predictions...")
pred=dtree.predict(data)

print("Creation du fichier predictions.txt...")

np.savetxt('predictions.txt',pred,delimiter="\t",fmt='%s')

print("Fin du programme")
end_time = (time.time() - start_time)
print("Temps d'execution : %d s" %int(end_time))