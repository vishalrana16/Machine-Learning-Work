import numpy as np
import pandas as pd

dataset = pd.read_csv('C:\\vishal\\study\\machine learning\\car_dataset.csv',header=None)

def rem(val):
    v =val
    if v == '5more' or v == 'more':
        return 5
    else:
        return v         

dataset[2] = dataset[2].apply(rem)
dataset[3] = dataset[3].apply(rem)


X = dataset.iloc[:,0:6].values
y = dataset.iloc[:,6].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label1 = LabelEncoder()
X[:,0] = label1.fit_transform(X[:,0])

label2 = LabelEncoder()
X[:,1] = label1.fit_transform(X[:,1])


label3 = LabelEncoder()
X[:,4] = label1.fit_transform(X[:,4])


label2 = LabelEncoder()
X[:,5] = label1.fit_transform(X[:,5])

label5 = LabelEncoder()
y[:] = label5.fit_transform(y[:])
onehot1 = OneHotEncoder(categorical_features = [0,1,4,5])
X = onehot1.fit_transform(X).toarray()


y = pd.to_numeric(y)
 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

def check(val):
    v = val
    if not pd.isnull(v):
        return 5        
    else:
        return v   
    
dataset[6].apply(check)



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,random_state = 1,
                                    max_features ='sqrt' ,
                                    criterion = 'entropy', 
                                    min_samples_split=29,
                                    bootstrap = True,
                                    #oob_score = True,
                                    verbose=0,
                                    #class_weight = balanced_subsample
                                    #warm_start = True)
                                    )
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)










