import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from seaborn import seaborn as sns  
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer.keys()
dict.keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
cancer.values()
dict.values(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

print(cancer['DESCR'])
print(cancer['target'])
print(cancer['target_names'])
['malignant' 'benign']
print(cancer['feature_names'])
cancer['data'].shape
(569, 30)
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'],['target']))
df_cancer.head()
df_cancer.tail()

sns.countplot(df_cancer['target'])
sns.scatterplot(x='mean area',y='mean smoothness',hue='target',data =df_cancer)

plt.figure(figsize =(20,10))
sns.heatmap(df_cancer.corr(), annot =True)

x = df_cancer.drop(['target'],axis =1)

x

y= df_cancer['target']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

y_pred = model.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
data = pd.read_csv("data/data.csv") 

data = data.drop(['Unnamed: 32', 'id'], axis=1)

data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
