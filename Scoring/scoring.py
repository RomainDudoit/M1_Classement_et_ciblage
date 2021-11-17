import numpy
import os
import pandas
os.chdir(os.path.dirname(__file__))
path = input("Chemin absolu du fichier à charger : ")
#-------------------------------------------------------------------------------------------------------
cols=['V14', 'V84', 'V96', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V197', 'V198', 'V199','V200']   
df =pandas.read_table(path,delimiter='\t',usecols=cols)

#encodage variable qualitative
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df[["V160","V161","V162"]]=df[["V160","V161","V162"]].apply(le.fit_transform)

#import modèle
import pickle
f=open("LDA.sav","rb")
lda=pickle.load(f)
f.close()

X = df.iloc[:,0:41]
y = df.iloc[:,41]
y=y.to_numpy().reshape(-1)


from sklearn import model_selection
XTrain,XTest,yTrain,yTest=model_selection.train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


#prediction
probas = lda.predict_proba(XTest)
print(probas)

classes=lda.classes_
print(lda.classes_)
score = probas[:,7]

#Insertion du scoring dans un DataFrame
dfScore= pandas.DataFrame(score,columns=['Score'])
dfScore.info()

#exportation dans un fichier excel
dfScore.to_csv("Score.csv",index=False)


#yTest en binaire
pos = pandas.get_dummies(yTest).values
#colonne positif
pos = pos[:,7]

#nb positifs(nb de m16)
npos = numpy.sum(pos)
print(npos)

#index pour tri selon le score croissant
index = numpy.argsort(score)
#inverser pour score décroissant
index = index[::-1]
#tri des individus (des valeurs 0/1)
sort_pos = pos[index]
#somme cumulée
cpos = numpy.cumsum(sort_pos)
#rappel
rappel = cpos/npos
#nb. obs ech.test
n = yTest.shape[0]
#taille de cible
taille = numpy.arange(start=1,stop=n+1,step=1)
#passer en pourcentage
taille = taille / n

#graphique
import matplotlib.pyplot as plt
plt.title('Courbe de gain')
plt.xlabel('Taille de cible')
plt.ylabel('Rappel')
plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(taille,taille,marker='.',color='blue')
plt.scatter(taille,rappel,marker='.',color='red')
plt.show()