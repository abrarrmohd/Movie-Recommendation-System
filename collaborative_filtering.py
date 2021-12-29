import pandas as pd
import numpy as np
import os
from numpy import linalg as LA
import math
import sys

train_path=sys.argv[1]
test_path=sys.argv[2]
df1 = pd.read_csv(train_path, header = None, names = ['Movie_Id','Cust_Id', 'Rating'])

df1['Rating'] = df1['Rating'].astype(int)

print("Train data--------------------------------------------------------------")
print(df1)

df_p = pd.pivot_table(df1,values='Rating',index='Cust_Id',columns='Movie_Id')
df_p = df_p.replace(np.NaN,0)
df_p=df_p.astype(int)

print("Train dataset-----------------------------------------------------------")
print(df_p)

ratings=df_p.to_numpy()

ratings_mean=np.divide(np.sum(ratings,axis=1),np.count_nonzero(ratings, axis=1))
ratings_mean=np.expand_dims(ratings_mean, axis = -1 )

mask = (ratings != 0)

ratings=np.where(mask, ratings-ratings_mean, 0)

ratings1=ratings/((np.expand_dims(LA.norm(ratings,axis=1),axis=-1))+1e-9)

w=np.dot(ratings1,ratings1.T)

print("Weight Matrix (Pearson Coefficient)--------------------------------------")
print(w)

#predicting values using knn from the weight matrix
pred = np.zeros(ratings.shape)
k=25
for i in range(ratings.shape[0]):
    sim_users = np.argsort(w[:,i])[-2:-k-2:-1]
    for j in range(ratings.shape[1]):
        pred[i, j] = np.dot(w[i, :][sim_users],(ratings[:, j][sim_users]))
        pred[i, j] /= (np.sum(np.abs(w[i, :][sim_users]))+1e-9)
pred=pred+ratings_mean

#filling predicted values into the dataframe
df_p1=pd.DataFrame(pred,index=df_p.index,columns=df_p.columns)

df_p1

#Test data
df2 = pd.read_csv(test_path, header = None, names = ['Movie_Id','Cust_Id', 'Rating'])

df2['Rating'] = df2['Rating'].astype(int)

print("Test Data----------------------------------------------------------------")
print(df2)

#RMSE
sum=0
for i in df2.index:
    pred_r=df_p1.loc[df2.loc[i,'Cust_Id'],df2.loc[i,'Movie_Id']]
    true_r=df2.loc[i,'Rating']
    sum=sum+((pred_r-true_r)**2)
print("RMSE Value= ",math.sqrt(sum/(i+1)))

sum=0
for i in df2.index:
    pred_r=df_p1.loc[df2.loc[i,'Cust_Id'],df2.loc[i,'Movie_Id']]
    true_r=df2.loc[i,'Rating']
    sum=sum+np.abs(pred_r-true_r)
print("MAE value=",sum/(i+1))
