# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:49:59 2022

@author: Bijen
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from Events import event
from matplotlib import pyplot as plt

def find_orientation(X,t):
    N = X.shape[0]
    X_new=np.zeros(X.shape)
    r_new=np.zeros([N])
    r_all=np.zeros([N])
    r_flip_all=np.zeros([N])
    label=np.zeros([N])
    for i in range(0,N):
        x=X[i]
        x_flip=x[::-1] #flip for S' pcc #comment for original one #can you a function
        x=np.reshape(x, t.shape)
        rho_t= np.corrcoef(x, t)
        r=rho_t[0,1]
        rho_flip=np.corrcoef(x_flip,t)
        r_flip= rho_flip[0,1]
        r_all[i]=r
        r_flip_all[i]=r_flip
        if r>r_flip:
            X_new[i]=x
            r_new[i]=r
            label[i]=1 #forward
        else:
            X_new[i]=x_flip #
            r_new[i]=r_flip
            label[i]=0 #backward
    return X_new, r_new, label, r_all, r_flip_all

def get_cluster_exampler_member(crit,X): # distance_i= pcc(exampler_i, examplar_k) for ith cluster
    N=X.shape[1]
    keysList = list(crit.keys())
    keysList = np.array(keysList)
    cluster_count=len(keysList)
    clusDict={};examDict={};distDict={}
    for i in range(0,cluster_count):
        cluster_members=list(crit.get(keysList[i]))
        member_count=len(cluster_members)
        X_cluster=np.zeros([member_count,N])
        for k in range(0,member_count):
            cluster_index=cluster_members[k]
            X_cluster[k,:]=X[cluster_index]
            cluster_i = pd.DataFrame(X_cluster)
            cluster_i.to_csv('Results/cluster' + str(i) + '.csv',header=False, index =False)
            clusDict[i] = X_cluster
            Examplar_i = X[keysList[i]]
            examDict[i] = Examplar_i
            distance_1_all=[]
            distance_2_all=[]
        for k in range(0,i): #for examplars other than i
            Examplar_k=X[keysList[k]]
            distance_1= (event.pcc(Examplar_i, Examplar_k)) # distance range 0,2, closer being 0.
            distance_1_all.append(distance_1)
        for k in range(i+1, cluster_count):
            Examplar_k=X[keysList[k]]
            distance_2= (event.pcc(Examplar_i,Examplar_k)) #distance_2= np.corrcoef(Examplar_i, Examplar_k)
            distance_2_all.append(distance_2)
        distance_all_i=distance_1_all+distance_2_all   #distance from other cluster di for i!=k, where k is the cluster
        distDict[i]= distance_all_i #smallest distance is the best or rank 1
    return examDict, clusDict, distDict

def get_cluster_rank(X,crit,t): # X=whole N spectra, ind=label from clustering forn N spectra
    keysList = list(crit.keys())
    cluster_count=len(keysList) 
    rank_score=np.zeros(cluster_count)
    norm_exp_k= np.zeros([cluster_count,np.size(X,1)])
    examDict, clusDict, distDict =get_cluster_exampler_member(crit,X) #get examplar and cluster member
    for k in range(0,cluster_count):
        r=[]
        X_ex = examDict.get(k)
        X_cluster= clusDict.get(k) 
        X_distance= distDict.get(k)
        avg = np.mean(X_cluster, axis=0)
        avg_spec = event.Event(0, avg)
        Exam_spec = event.Event(0, X_ex)
        theory_spec = event.Event(0, t[0,:])
        norm_theory = event.normalize(theory_spec.data)
        norm_exam = event.normalize(Exam_spec.data)
        norm_exp = event.normalize(avg_spec.data)
        norm_exp_k[k]=norm_exp
        pcc_result_exam=event.pcc(norm_theory,norm_exam)
        pcc_result_exam=str(round(pcc_result_exam, 3))
        pcc_result_avg=event.pcc(norm_theory, avg_spec.data)
        pcc_result_avg=str(round(pcc_result_avg, 3))
        avg_spec_moving= event.moving_averages(norm_exp, 24) 
        pcc_result_mov=event.pcc(norm_theory,avg_spec_moving)
        pcc_result_mov=str(round(pcc_result_mov, 3))
        fig= plt.figure()
        fig.set_figwidth(20)
        fig.set_figheight(10)
        ax=plt.axes()
        ax.set(facecolor = "white")
        plt.rcParams.update({'font.size': 12})
        plt.plot(norm_theory, label="Theoretical")
        plt.plot(norm_exam, label="Examplar")
        plt.plot(avg_spec_moving, label="Cluster_moving_average")
        plt.plot(norm_exp, label="Cluster Average")
        plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.3,0.9), ncol=2)
        plt.xlabel("Time")
        event_fname= "Cluster members="+str(len(X_cluster))+" cluster_"+str(k)
        plt.ylabel("Normalized signal Cluster_"+str(event_fname))
        plt.title("Final PCC_value from examplar= %s cluster average=%s and moving average =%s" %(pcc_result_exam,pcc_result_avg,pcc_result_mov ))
        plt.grid(False)
        plt.savefig('Results/Sim_Dist_graph_{}.jpg'.format(k))
        plt.tight_layout()
        plt.show() 
        plt.close()           
        for i in range(0,len(X_cluster)):
            r_score=event.pcc(X_cluster[i],X_ex) # avg for X_ex X_cluster[i] is the ith cluster the cluster members and instead X_examplar we can use X_cluster average
            r.append(r_score)
        rank_score[k]= np.mean(r)/np.mean(X_distance) 
    # temp =(-rank_score).argsort()[:len(rank_score)] #highest score rank 1
    temp = rank_score.argsort() #lowest score rank 1
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(rank_score))
    
    pcc_result_cluster= np.zeros([cluster_count])
    for k in range(0,cluster_count):
        pcc_result_cluster[k]= event.pcc(norm_theory, norm_exp_k[k])
    temp1 = (-pcc_result_cluster).argsort()[:len(rank_score)]
    ranks_pcc = np.empty_like(temp1)
    ranks_pcc[temp1] = np.arange(len(pcc_result_cluster))
    fig= plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    plt.rcParams.update({'font.size': 12})
    for k in range(0,cluster_count):
        cluster_name= 'cluster_'+str(k)
        cluster_name_all= cluster_name +' True PCC='+str(round(pcc_result_cluster[k],2))+' pcc_rank='+str(ranks_pcc[k]+1) + ' Proposed ranking='+str(ranks[k]+1)
        plt.plot(norm_exp_k[k], label=cluster_name_all)
        plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.3,1.2), ncol=2)
    plt.savefig('Results/All_rank.pdf')
    plt.show()
    plt.tight_layout()
    return rank_score, pcc_result_cluster, ranks+1, ranks_pcc+1
def sil_cluster(X,ind):
    unique_ind= np.unique(ind)
    new_ind=[]
    silhouette_avg = silhouette_score(X, ind)
    silhouette_values_samples = silhouette_samples(X, ind)
    silhouette_samples_cluster=np.zeros([len(unique_ind)])
    for k in range(0, len(unique_ind)):
        test=[]
        test_all=[]
        for i in range (0, len(ind)):
               if ind[i]==unique_ind[k]:
                   test=silhouette_values_samples[i]
                   test_all.append(test)
                   # print(test)
               else:
                   print('not matched')
               new_ind.append(k)
        silhouette_samples_cluster[k]=np.mean(test_all)
    return(silhouette_samples_cluster, silhouette_avg)

def get_plot(X,crit,ind,t):
    [Xnew, rnew, label, r_all, r_flip_all]= find_orientation(X,t)
    k=0
    keysList = list(crit.keys())
    cluster_count=len(keysList)
    for fig_number in range(0,cluster_count):
        x_f=[]
        y_f=[]
        label_f=[]
        for i in range(0, len(X)): 
            if (ind[i]==keysList[k]):# manually label ko value change garnu parxa #95	89	332	457	415 pcc_d#64	343	40	217	195	230	412	85	421	237	447	124	391
                x_f.append(r_all[i])
                y_f.append(r_flip_all[i])
                label_f.append(label[i])
            else:
                print(i)
        k=k+1
        print('loop finished here')
        print(len(label_f))
        ax=plt.axes()
        ax.set(facecolor = "white")
        scatter=plt.scatter(x_f,y_f,c=label_f, cmap='rainbow', alpha=0.7, edgecolors='b') # manually get plot for each cluster
        plt.title("Total spectra in Clusters ="+str(len(label_f)))    
        plt.xlabel("PCC(S,T)")
        plt.ylabel("PCC(S_flipped,T") 
        plt.legend(*scatter.legend_elements(), loc="upper right", title="Orientation")
        plt.savefig('Results/Sim_PCC_scatter_{}.jpg'.format(fig_number))
        plt.grid(False)
        plt.show()
        plt.close()
