# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:59:04 2022

@author: Bijen
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import csv
sys.path.insert(0,'Events')
from Events import my_event
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
# args = sys.argv[1:]
# if (len(args) == 0):
#     raise Exception("Please pass in the filename of your data")


# theory_fname = open(sys.argv[1], "r")
# event_fname = open(sys.argv[2], "r")
# Examplar_fname = open(sys.argv[3], "r")
# cluster_fname = open(sys.argv[4], "r")
# ind_fname = open(sys.argv[5], "r")


# a1 = r"202212_nanospectra_code/oxa_181/lt119/theory_trace_2000.csv"
# a2 = r"202212_nanospectra_code/oxa_181/lt119/event.csv"
# a3 = r"Results/Oxa_181/First 500/Moving average/PCC_new/examplar.csv"
# a4 = r"Results/Oxa_181/First 500/Moving average/PCC_new/Cluster.csv"
# a5 = r"Results/Oxa_181/First 500/Moving average/PCC_new/ind.csv"

a1 = r"202212_nanospectra_code/low_freq_AB42/theory_trace.csv"
a2 = r"202212_nanospectra_code/low_freq_AB42/ab42_events.csv"
a3 = r"Results/Low_ab42/PCC_new/examplar.csv"
a4 = r"Results/Low_ab42/PCC_new/Cluster.csv"
a5 = r"Results/Low_ab42/PCC_new/ind.csv"

# a1 = r"202212_nanospectra_code/high_freq_AB42/pm13/theory_trace_1000.csv"
# a2 = r"202212_nanospectra_code/high_freq_AB42/pm13/event.csv"
# a3 = r"Results/High_ab42/First 500/Moving averages/PCC_new/examplar.csv"
# a4 = r"Results/High_ab42/First 500/Moving averages/PCC_new/Cluster.csv"
# a5 = r"Results/High_ab42/First 500/Moving averages/PCC_new/ind.csv"

t = pd.read_csv(a1, header=None)
t=t.to_numpy()
X= pd.read_csv(a2, header=None)
X = X.head(500)
X=X.to_numpy()
Ex = pd.read_csv(a3, header=None)
Ex=Ex.to_numpy()
Cl_pd= pd.read_csv(a4, header=None)
ind= pd.read_csv(a5, header=None)
ind=ind.to_numpy()
Cl = []
for c in range(0, Cl_pd.shape[1]):
    val = Cl_pd[c][0]
    temp = val.split('[')[1].split(']')[0].split(',')
    temp_arr =[]
    for test in temp:
        temp_arr.append(int(test))
    Cl.append(temp_arr)
crit={}
for i in range(0, Ex.shape[1]):
    key= (Ex[0,i])
    value= Cl[i] # this value to be interger
    crit[key] = value
print(crit)

ind=ind.transpose()
# rank_score, pcc_score, ranks_ours, ranks_pcc = my_event.get_cluster_rank(X, crit, t)
my_event.get_plot(X,crit,ind,t)
# sil_score= my_event.sil_cluster(X,ind)
# temp = -sil_score[0].argsort() #lowest score rank 1
# ranks_sil = np.empty_like(temp)
# ranks_sil[temp] = np.arange(len(sil_score[0]))
# ranks_sil= ranks_sil+1
print('hello crit')
# [Xnew, rnew, label, r_all, r_flip_all]= my_event.find_orientation(X,t)
# DB_rank_my= my_event.DB_rank(X,crit)
CH_score=calinski_harabasz_score(X,ind)
DB_scores= davies_bouldin_score(X,ind)
# DB_scores_true = davies_bouldin_score(X,label)
