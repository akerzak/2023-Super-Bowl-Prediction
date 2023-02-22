# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:43:41 2023

@author: Andrew's Laptop II
"""
#load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

#read in historical dataset
superbowldata = pd.read_csv(r'C:\path\to\historical_superbowl_data.csv')
print(superbowldata) 

descriptsb = superbowldata.describe()
print(descriptsb)

corr = superbowldata.corr()
plt.subplots(figsize = (20,15))
heatmap=sns.heatmap(corr,xticklabels = corr.columns, yticklabels = corr.columns, annot = True)

y = superbowldata.hwin_sb
x = superbowldata.drop(columns = ['hwin_sb'])            #prepare x_train data without any quality data

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=50)

# Define the naive Bayes Classifier
model = MultinomialNB()

# Train the model 
model.fit(xtrain, ytrain)

# Predict Output using the test data
pred = model.predict(xtest)

# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)

plt.subplots(figsize = (20,15))
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')

# store an array of predicted and actual labels
d = {'win_sb':ytest, 'prediction':pred}

# turn the array into a data frame and print it
output = pd.DataFrame(data=d)
output

scores = cross_val_score(model,xtrain,ytrain)
scores.mean()
scores.std()

ex = pd.read_csv(r'C:\path\to\2023_teams_data.csv')
p = model.predict(ex)
print(p)

#knn classifier 
features=list(zip(superbowldata['hleaguerank_pointsfor_offense'],superbowldata['hleaguerank_pointsagaist_defense'],superbowldata['hleaguerank_turnovers_of'],superbowldata['hleaguerank_turnovers_df'],superbowldata['hsrs'],superbowldata['leaguerank_pointsfor_offense'],superbowldata['leaguerank_pointsagaist_defense'],superbowldata['leaguerank_turnovers_of'],superbowldata['leaguerank_turnovers_df'],superbowldata['srs'],superbowldata['hwin_sb']))

modelknn = KNeighborsClassifier(n_neighbors=1)

# Train the model using the training sets
modelknn.fit(xtrain,ytrain)


#create place to store error rates from loop
error_rates = []
#loop to find the 'k' with the lowest error 
for i in np.arange(1, 17):

    new_model = KNeighborsClassifier(n_neighbors = i)

    new_model.fit(xtrain, ytrain)

    new_predictions = new_model.predict(xtest)
    
    error_rates.append(np.mean(new_predictions != ytest))
#plot to view ks confirm that 1k is the lowest error rate 
plt.subplots(figsize = (20,15))
plt.plot(error_rates)  

scoresknn = cross_val_score(modelknn,xtrain,ytrain)
scoresknn.mean()
scoresknn.std()

# now we give the model some input and see what it predicts
predicted = modelknn.predict(ex)
print(predicted)