# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:59:04 2022

@author: Bijen
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
sys.path.insert(0,'Events')
from Events import my_event
# args = sys.argv[1:]
# if (len(args) == 0):
#     raise Exception("Please pass in the filename of your data")


theory_fname = open(sys.argv[1], "r")
print("theory file name", theory_fname)
event_fname = open(sys.argv[2], "r")
Examplar_fname = open(sys.argv[3], "r")
cluster_fname = open(sys.argv[4], "r")
ind_fname = open(sys.argv[5], "r")

# a1 = r"202212_nanospectra_code/low_freq_AB42/theory_trace.csv"
# a2 = r"202212_nanospectra_code/low_freq_AB42/ab42_events.csv"
# a3 = r"Results/Low_ab42/PCC_D/examplar.csv"
# a4 = r"Results/Low_ab42/PCC_D/Cluster.csv"
# a5 = r"Results/Low_ab42/PCC_D/ind.csv"

t = pd.read_csv(theory_fname, header=None)
t=t.to_numpy()
X= pd.read_csv(event_fname, header=None)
X=X.to_numpy()
Ex= pd.read_csv(Examplar_fname, header=None)
Ex=Ex.to_numpy()
Cl= pd.read_csv(cluster_fname, header=None)
Cl=Cl.to_numpy()
ind= pd.read_csv(ind_fname, header=None)
ind=ind.to_numpy()
crit={}
i=0
while i<np.size(Ex):
    key= int(Ex[0,i])
    value= int(Cl[0,i])
    crit[key] = value 
    i=i+1
print(crit)
# t_z = stats.zscore(np.transpose(t)) 
rank_score, pcc_score, ranks_ours, ranks_pcc = my_event.get_cluster_rank(X, crit, t)
sil_score= my_event.sil_cluster(X,ind)
print('hello crit')
[Xnew, rnew, label, r_all, r_flip_all]= my_event.find_orientation(X,t)
my_event.get_plot(X,crit,ind,t)
