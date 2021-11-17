#%% Importing Libraries
import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
os.chdir(os.path.dirname(__file__))

#%% Importing Data
data = pd.read_csv("./data/data_avec_etiquettes.txt", sep="\t")

encoder = OrdinalEncoder()
encoder.fit(data[["V160", "V161", "V162", "V200"]])
data[["V160", "V161", "V162", "V200"]] = encoder.transform(data[["V160", "V161", "V162", "V200"]])

X = data.iloc[:,0:199]
y = data.iloc[:,199]
print(data.head())

#%% 1. Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols) > 0):
    p = []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y.astype(float), X_1.astype(float)).fit()
    p = pd.Series(model.pvalues.values[1:], index=cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
    
selected_features_BE = cols
print(selected_features_BE)

# Columns to keep by section 1 : 
# ['V14', 'V84', 'V96', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165',
# 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175',
# 'V176', 'V177', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187',
# 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V197', 'V198', 'V199']

#%% 2.1 Removing constant features
columns_to_remove = set()

constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(X)
constant_columns = [column for column in X.columns if column not in X.columns[constant_filter.get_support()]]

columns_to_remove.update(constant_columns)
X = X.drop(columns=constant_columns)
print(X.shape)
    
#%% 2.2 Removing duplicated features
X_d = X.T.duplicated()
duplicated_columns = list(X_d[X_d==True].index)

columns_to_remove.update(duplicated_columns)
X = X.drop(columns=duplicated_columns)
print(X.shape)

#%% 2.3 Removing correlated features
num_columns = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(X.select_dtypes(include=num_columns).columns)
X_numonly = X[numerical_columns]

correlated_features = set()
correlation_matrix = X_numonly.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            
columns_to_remove.update(correlated_features)
X = X.drop(columns=correlated_features)

# Columns to remove by section 2 :
# V194,V161,V173,V191,V198,V196,V195,V174,V178,V182,V197,V199
# V186,V179,V169,V167,V165,V188,V192,V183,V176,V172,V177,V187
# V184,V181,V175,V180

#%% 3. SelectKBest

bestfeatures = SelectKBest(score_func=f_classif, k=100) # We set a custom k value, so here we want 100 features

fit = bestfeatures.fit(X, y)

dfscores = pd.DataFrame(fit.pvalues_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
featureScores = featureScores[featureScores['Score']>0.05] # We only take values greater than 5%
#print(featureScores.nlargest(140, 'Score'))
print(featureScores)

#%% 4. ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X, y)
importances = pd.DataFrame({'label':X.columns.array.to_numpy(), 'score':model.feature_importances_}, columns=['label', 'score'])
print(importances.nlargest(30, 'score'))

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
