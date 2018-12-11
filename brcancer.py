# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import datasets
#from sklearn.datasets import load_breast_cancer
#data = load_breast_cancer()

# Import data
df = pd.read_csv("C:/Users/dmunene/OneDrive - Dalberg Global Development Advisors/RESSOL/Personal/Data Analysis/SatoAI/Classification/Challenge2/wisconsin_breast_cancer.csv")

# Checking for nas
df.isnull().sum()

# Filling nas
df = df.fillna(df.mean())

df.columns

df1 = df.drop('class', axis = 1)
y = df['class'].values
X = df1[:].values



# Splitting into test and train

(X_train,X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.2, random_state = 1)

#plt.scatter(data.feature_names[:,0], data.feature_names[:,1], s = 50, cmap = 'RdBu')



model = GaussianNB()
model.fit(X_train,y_train);

outpt = model.predict(X_test)
outpt = pd.DataFrame(outpt)


# assessing the accuracy
accuracy_score(y_test,outpt)

# Modelling in RandomForest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 1000,max_features = 10,oob_score = True,random_state = 301)

rf_model.fit(X_train,y_train)

output2 = rf_model.predict(X_test)
output2 = pd.DataFrame(output2)

rf_model.oob_score_

#accuracy_score(y_test,output2)

