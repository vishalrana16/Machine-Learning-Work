import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('criminal_train.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)
X_test = sc.transform(X_test)
y_train = y

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20,random_state=1,
                                    max_features ='auto' ,
                                    criterion = 'gini', 
                                    min_samples_split=30,
                                    bootstrap = True,
                                    oob_score = True,
                                    verbose=2,
                                    warm_start = True
                                    
                                    )
hist = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


features = pd.DataFrame()
features['feature'] = dataset.columns[:71]
features['importance'] = classifier.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))


feat = pd.DataFrame()
feat['feature'] = dataset.columns[:71]

feat['importance'] = classifier.feature_importances_
feat.sort_values(by=['importance'], ascending=False, inplace=True)

feat = feat.head(25)

cf = list(feat['feature'])

print(cf)

dataset = pd.read_csv('criminal_train.csv')
df = dataset[['IRFAMIN3',
 'POVERTY3',
 'IRPINC3',
 'PRVHLTIN',
 'IFATHER',
 'GRPHLTIN',
 'IRPRVHLT',
 'ANALWT_C',
 'PERID',
 'VESTR',
 'IRMEDICR',
 'OTHINS',
 'CAIDCHIP',
 'PRXYDATA',
 'IRHHSIZ2',
 'IRKI17_2',
 'NRCH17_2',
 'IRHH65_2',
 'IRFAMSOC',
 'MEDICARE',
 'IRMCDCHP',
 'IROTHHLT',
 'HLTINNOS',
 'CELLNOTCL',
 'IRFSTAMP',
  'Criminal' ]]

X = df.iloc[:,:-1].values
y = df.iloc[:, -1].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)
X_test = sc.transform(X_test)
y_train = y

classifier = RandomForestClassifier(n_estimators = 20,random_state=1,
                                    max_features ='auto' ,
                                    criterion = 'gini', 
                                    min_samples_split=30,
                                    bootstrap = True,
                                    oob_score = True,
                                    verbose=2,
                                    warm_start = True
                                    )
hist = classifier.fit(X_train, y_train)



y_sol = classifier.predict(X_test)

df_test = pd.read_csv('criminal_test.csv')
df_test = df_test[['IRFAMIN3',
 'POVERTY3',
 'IRPINC3',
 'PRVHLTIN',
 'IFATHER',
 'GRPHLTIN',
 'IRPRVHLT',
 'ANALWT_C',
 'PERID',
 'VESTR',
 'IRMEDICR',
 'OTHINS',
 'CAIDCHIP',
 'PRXYDATA',
 'IRHHSIZ2',
 'IRKI17_2',
 'NRCH17_2',
 'IRHH65_2',
 'IRFAMSOC',
 'MEDICARE',
 'IRMCDCHP',
 'IROTHHLT',
 'HLTINNOS',
 'CELLNOTCL',
 'IRFSTAMP'
   ]]

X = df_test.values


sc1 = StandardScaler()
X = sc.transform(X)

y_sol = classifier.predict(X)




df_output = pd.DataFrame()

df_output['PERID'] = df_test['PERID']
df_output['Criminal'] = y_sol
df_output[['PERID','Criminal']].to_csv('criminal_new_3.csv',index=False)

