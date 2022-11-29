# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:19:20 2022

@author: Bijen K
"""

#AP python file
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import statistics
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
def cl_extra(ind,k):
    new_ind=ind.reshape(-1,1) #Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
    k=k
    kmeans = KMeans(k)
    new_ind= kmeans.fit(new_ind)
    return new_ind
   
class Cluster:
  def __init__(self, ava, res, exm):
    self.ava = ava
    self.res = res
    self.exm = exm

  def get_C(self):#identify examplar and assosciated datapoints
      A=self.ava
      R=self.res
      C=self.exm
      row=[]
      max_value_all=np.zeros([N])
      index=np.zeros([N])
      for i in range(0,N):
          for k in range(0,N):
              C[i,k]=A[i,k]+R[i,k]
      for i in range(0,N):
          row= (C[i])# finding exampler from each row with index
          row=row.tolist()
          max_value = max(row)
          max_value_all[i]=max_value
          # max_value_all.append(max_value) only for list
          index[i] = row.index(max_value)   
      return C,index,max_value_all
def get_S(dfn):
    data=dfn
    datarr=data.to_numpy()
    S_new=np.zeros([N,N])
    z=[]
    for row_idx in range(0,N):
        for col_idx in range(1,N):
                # for i in range(0, N2):
                    row1= datarr[row_idx]
                    row2= datarr[col_idx]
                    z=(row1-row2)*(row1-row2)
                    S_new[row_idx, col_idx]= -sum(z)
                    S_new[col_idx, row_idx]=S_new[row_idx, col_idx]
                    # print(row_idx,col_idx)
    dia = S_new.min()
    median = np.median(S_new)
    i=0
    while i<N:
            S_new[i][i] = median # median later
            i += 1
    return S_new
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
# rng=random(1)
df = pd.read_csv(r"C:\Users\Bijen\Documents\AffinityPropagation-master\lt076_event.csv",header=None) #need to add a 1st row with title or nothing, data only from 2nd row
data=df
# data=df[df.columns[0:200]] #for columns/features
# data = data.head(10) #for rows/sample
# = load_iris(return_X_y=True)
# X.shape=(100,4)
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y) # feature selection are mostly done with labels 'y' so not applicable here. Eg. recursive feature elemination (rfe)
# X_new.shape= (100,2)
# X_rs = data.copy()
# sc = StandardScaler()
# for c in X_rs.columns:
#     X_rs[c] = sc.fit_transform(X_rs[c].values.reshape(-1,1))
# data=X_rs
N = data.shape[0]
N2=data.shape[1]
S = np.zeros([N,N])
R = np.zeros([N,N])
A = np.zeros([N,N])
C = np.zeros([N,N])
iter = 100
lambdax = 0.5
datarr=data.to_numpy()
color_map= np.zeros([N,N2])
E=[]
all_cc=[]
m=0
same_count=0
S=get_S(data)
print('Similarity matrix done')
temp = np.ones([N,N])
while m<iter:
    R = get_R(S,A) #update responsibility
    A = get_A(R,A) #update availibility
    Ex = Cluster(A,R,C) #update Criteron 
    [E,ind, maxval]=Ex.get_C()
    cc=len(np.unique(ind))
    if ((temp==ind).all()):
        same_count=same_count+1;
        print('same decision')
    else:
        same_count=same_count;
    if (same_count>=10):
        print('Convergence achieved at m=',m)
        break;
    else:
        temp = ind.copy()
        print('Continued iteration at m=',m)
        m=m+1 #need to define stopping criteria
    all_cc.append(cc)
    print('No of clusters identified=',cc)
    
print('completed iteration=',m)
print(ind)
k=2 #0 start
ind_kmeans=cl_extra(maxval,k)
ind_k=ind_kmeans.labels_
color_map = get_map(ind, data)
X=data.to_numpy()
plt.figure()
# area = (30 * np.random.rand(N))**2
# plt.scatter(X[:,0], X[:,1],s= area, c=ind, cmap='Paired')
plt.scatter(X[:,0], X[:,1],c=ind, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title("Clusters determined by Affinity with clusters="+str(cc))
plt.show()
plt.close()
# colors = np.random.rand(N)
plt.figure()
# area = (30 * np.random.rand(N))**2
# plt.scatter(X[:,0], X[:,1],s= area, c=ind, cmap='Paired')
plt.scatter(X[:,0], X[:,1],c=ind_k, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title("Clusters determined by K-means Affinity at k="+str(k))
plt.show()
plt.close()

area = (30 * np.random.rand(N,N2))**2 
plt.figure()
plt.scatter(X,color_map,s=area)
plt.title("Clusters determined by Color-map Affinity")
plt.show()
plt.close()

plt.plot(all_cc, lw=1)
plt.xlabel('Number of Iteration')
plt.ylabel('Number of clusters')
plt.title('ab42_events.csv'+str(data.shape))  
plt.show()
plt.close()
#%%
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import scipy.stats
n_com=2
# X_pls= PLSRegression(n_components=2).transform(X)
X_pca= PCA(n_components=n_com).fit_transform(X) #In case of uniformly distributed data, LDA almost always performs better than PCA. However if the data is highly skewed (irregularly distributed) then it is advised to use PCA since LDA can be biased towards the majority class.
X_iso= Isomap(n_components=2).fit_transform(X)
X_tsne = TSNE(n_components = n_com).fit_transform(X)
# X_lda=LDA(n_components=1).transform(X)
plt.scatter(X_pca[:,0], X_pca[:,1],c=ind, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('PCA ab42_events.csv'+str(X_pca.shape))
plt.show()
plt.scatter(X_iso[:,0], X_iso[:,1],c=ind, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('ISO ab42_events.csv'+str(X_iso.shape))
plt.show()
plt.scatter(X_tsne[:,0], X_tsne[:,1],c=ind, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('TSNE ab42_events.csv'+str(X_tsne.shape))
plt.show()
# plt.scatter(X_lda[:,0], X_lda[:,1],c=ind, cmap='rainbow', alpha=0.7, edgecolors='b')
# plt.title('PLS ab42_events.csv'+str(X_lda.shape))
# r_value=scipy.stats.pearsonr(X,X_pca)
#%% # r(i,k)=s(i,k)-max{a(i,k'+s(i,k'))} k'!=k         
