# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:19:20 2022

@author: Bijen K
"""

import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from Events import my_event
from Events import event
import csv
# N is the number of two-dimension data points
# N is the number of two-dimension data points
# S is the similarity matrix
# R is the responsibility matrix
# A is the availabiltiy matrix
# iter is the maximum number of iterations
# lambda is the damping factor      

def get_map(ind, data):
    cl_map=np.zeros([N,N2])
    for i in range(0,N):
         for k in range(0,N2):
             cl_map[i,k]=ind[i]
    return cl_map
def cl_extra(maxval,k):
    new_ind=maxval.reshape(-1,1) #Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
    kmeans = KMeans(k)
    new_ind= kmeans.fit(new_ind)
    return new_ind
   
class Cluster:
  def __init__(self, ava, res, crit):
    self.ava = ava
    self.res = res
    self.crit = crit

  def get_C(self):#identify examplar and assosciated datapoints
      A=self.ava
      R=self.res
      C=self.crit
      row=[]
      max_value_all=np.zeros([N])
      cl_index=np.zeros([N])
      for i in range(0,N):
          for k in range(0,N):
              C[i,k]=A[i,k]+R[i,k]
      critDict = {}
      for i in range(0,N):
          row= (C[i])# finding exampler from each row with index
          row=row.tolist()
          max_value = max(row)
          max_value_all[i]=max_value
          # max_value_all.append(max_value) #only for list
          cl_index[i] = row.index(max_value) 
          key = np.argmax(C[i])
          if key not in critDict:
              critDict[key] = [i]
          else:
              critDict[key].append(i)
      return C, critDict, cl_index, max_value_all 

def get_S(dfn):
    S_new=np.zeros([N,N])
    z=[]
    for row_idx in range(0,N):
        for col_idx in range(0,row_idx):
            row1= dfn[row_idx]
            row2= dfn[col_idx]
            z=(row1-row2)*(row1-row2)
            S_new[row_idx, col_idx]= -sum(z)
            S_new[col_idx, row_idx]=S_new[row_idx, col_idx]                  
    i=0
    while i<N:
            S_new[i][i] = S_new.min() # median later np.median(S_new)
            i += 1
    return S_new

def get_pcc(X):
    rho_x=np.zeros([N,N])
    d_x=np.zeros([N,N])
    d_new=np.zeros([N,N])
    for i in range(0,N):
        for k in range(0,i):
            x=X[i]; y=X[k]
            r=np.corrcoef(x,y)
            x_flip=x[::-1]
            e1= np.corrcoef(x,y)
            e2= np.corrcoef(x_flip,y)
            # e3= np.corrcoef(x,y_flip)
            rho_x[i,k]=r[0,1]-1
            rho_x[k,i]=rho_x[i,k]
            d_x[i,k] = 2-e1[0,1] + e2[0,1] #e2= X[k], e1_rev= X[::-1] d(e1,e2)=2-PCC(e1, e2) +PCC(reverse(e1),e2) 
            d_x[k,i] = d_x[i,k] # 2- e1[0,1] + e3[0,1]
            d_new[i,k] = e1[0,1] - e2[0,1]-2 # [-2,0] -1/(e1[0,1]+e2[0,1])#-12/(e1[0,1]+e2[0,1]+2)+3
            d_new[k,i]=d_new[i,k]                          
    i=0
    while i<N:
             rho_x[i,i] = rho_x.min() # median later
             d_x[i,i] = d_x.min()
             d_new[i,i] = d_new.min()
             i += 1
    return rho_x,d_x,d_new
def get_R(S,A): #r(i,k)=s(i,k)-max{a(i,k')+s(i,k'))} k'!=k
    S=S
    A=A
    for i in range(0,N):
        for k in range(0,N):
            max_buff=-1e100000000000000
            for kk in range(0,k):
                if (A[i,kk]+S[i,kk])>max_buff:
                    max_buff= (A[i,kk]+S[i,kk])
            for dk in range(k+1,N):
                if (A[i,dk]+S[i,dk])>max_buff:
                    max_buff= (A[i,dk]+S[i,dk])
            R[i,k] = (1-lambdax)*(S[i,k]-max_buff)+lambdax*R[i,k]
    return R
def get_A(R,A):# a(i,k)= min{0, r(k,k)+ sum(max(0, r(i',k)))} i'!=i
    R=R
    A=A
    for i in range(0,N):
        for k in range(0,N):
            sum_value=0
            if i==k:
                for ii in range(0,i):
                    if (R[ii,k])>0:
                        sum_value= sum_value+R[ii,k]
                    else:
                        sum_value = sum_value+0
                for ii in range(i+1,N):
                    if (R[ii,k])>0:
                        sum_value= sum_value+R[ii,k]
                    else:
                        sum_value = sum_value+0 # [i,j]=sum[]- R[ii,j]
                A[i,k]=(1-lambdax)*sum_value + lambdax*A[i,i]
            else:
                sum_value=0
                for ii in range(0,i):
                    if (R[ii,k])>0:
                        sum_value= sum_value+R[ii,k]
                    else:
                        sum_value = sum_value+0
                for ii in range(i+1,N):
                    if (R[ii,k])>0:
                        sum_value= sum_value+R[ii,k]
                    else:
                        sum_value = sum_value+0 # [i,j]=sum[]- R[ii,j]
                a_value= R[k,k]+sum_value
                min_value= min(0,a_value)
                A[i,k]=(1-lambdax)*min_value + lambdax*A[i,k]
            # print(A) # to check changing A
    return A 

def run(Simmat):
    temp =  {}
    E=[];all_cc=[];same_count=0;m=0
    A = np.zeros([N,N])
    C = np.zeros([N,N])
    while m<iter:
        R = get_R(Simmat,A) #update responsibility
        A = get_A(R,A) #update availibility
        Ex = Cluster(A,R,C) #update Criteron 
        [E, crit, ind, maxval]=Ex.get_C()
        cc=len(crit)
        if (sorted(temp.values()) == sorted(crit.values())):
            same_count=same_count+1;
            print('same decision')
        else:
            same_count=same_count
        if (same_count>=10):
            print('Convergence achieved at m=',m)
            break;
        else:
            temp = crit.copy()
            print('Continued iteration at m=',m)
            m=m+1 #need to define stopping criteria
        all_cc.append(cc)
        print('completed iteration=',m)
        print('No of clusters identified=',cc)
        C=E
    return crit,cc, ind, maxval
# def main():
fname=r'202212_nanospectra_code/oxa_181/lt119/event.csv'
df = pd.read_csv(fname,header=None) #need to add a 1st row with title or nothing, data only from 2nd row
data=df
# data=df[df.columns[0:200]] #for columns/features
data = data.head(500) #for rows/sample
data=data.to_numpy()
X=data
N = data.shape[0]
N2=data.shape[1]
# X_flip = np.zeros([N,N2])
# X_moving = np.zeros([N,N2])
# # X_norm = np.zeros([N,N2])
# for i in range(0, len(X)):
#     exp_data_i= X[i]
#     X_moving[i]= event.moving_averages(exp_data_i, 48)
#     X_norm[i]= event.normalize(exp_data_i)
iter = 100
lambdax = 0.5
R = np.zeros([N,N])
color_map= np.zeros([N,N2])
# S=get_S(X)
# print('Similarity matrix done')
[PCC, PCC_D, PCC_new] = get_pcc(X)
print('PCC matrix done')
crit,cc,ind, maxval= run(PCC_new) # change here S or PCC
with open('Results/Cluster.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(crit.values())
with open('Results/ind.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(ind)
with open('Results/examplar.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(crit.keys())
