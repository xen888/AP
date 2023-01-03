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
            # cluster_i = pd.DataFrame(X_cluster)
            # cluster_i.to_csv('Results/cluster' + str(i) + '.csv',header=False, index =False)
            clusDict[i] = X_cluster
            Examplar_i = X[keysList[i]]
            examDict[i] = Examplar_i
            distance_1_all=[]
            distance_2_all=[]
        for k in range(0,i): #for examplars other than i
            Examplar_k=X[keysList[k]]
            distance_1= (event.pcc(Examplar_i, Examplar_k))+1 # distance range 0,2, closer being 0.
            distance_1_all.append(distance_1)
        for k in range(i+1, cluster_count):
            Examplar_k=X[keysList[k]]
            distance_2= (event.pcc(Examplar_i,Examplar_k))+1 #distance_2= np.corrcoef(Examplar_i, Examplar_k)
            distance_2_all.append(distance_2)
        distance_all_examplar=distance_1_all+distance_2_all   # here I can do filtering  distance from other cluster di for i!=k, where k is the cluster
        distDict[i]= distance_all_examplar #smallest distance is the best or rank 1
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
        X_ex_flipped= X_ex[::-1]
        X_cluster= clusDict.get(k)
        X_cluster_clean = clean_cluster(X_cluster,X_ex,t,k)
        X_distance= distDict.get(k)
        avg_cluster = np.mean(X_cluster, axis=0)
        avg_X= np.mean(X, axis=0)
        avg_cluster_clean= np.mean(X_cluster_clean, axis=0)
        avg_spec = event.Event(0, avg_cluster)
        avg_spec_clean = event.Event(0, avg_cluster_clean)
        Exam_spec = event.Event(0, X_ex)
        theory_spec = event.Event(0, t[0,:])
        norm_theory = event.normalize(theory_spec.data)
        norm_theory_moving = event.moving_averages(norm_theory, 24)
        # norm_exam = event.normalize(Exam_spec.data)
        norm_exp = event.normalize(avg_spec.data)
        norm_exp_clean= event.normalize(avg_spec_clean.data)
        norm_exp_k[k]=norm_exp
        # pcc_result_exam=event.pcc(norm_theory,norm_exam)
        # pcc_result_exam=str(round(pcc_result_exam, 3))
        # pcc_result_avg=event.pcc(norm_theory, avg_spec.data)
        # pcc_result_avg=str(round(pcc_result_avg, 3))
        avg_spec_moving_clean= event.moving_averages(norm_exp_clean, 24) 
        avg_spec_moving= event.moving_averages(norm_exp, 24) 
        pcc_result_mov=event.pcc(norm_theory_moving,avg_spec_moving)
        pcc_result_mov=str(round(pcc_result_mov, 3))
        pcc_result_mov_clean= str(round(event.pcc(norm_theory_moving,avg_spec_moving_clean),3))
        # avg_spec_moving_flipped= event.moving_averages(norm_exp_clean[::-1], 24)
        pcc_result_mov_clean_flipped = str(round(event.pcc(norm_theory_moving,avg_spec_moving),3))
        fig= plt.figure()
        fig.set_figwidth(20)
        fig.set_figheight(10)
        ax=plt.axes()
        ax.set(facecolor = "white")
        plt.rcParams.update({'font.size': 12})
        plt.plot(norm_theory_moving, label="Moving Theoretical")
        # plt.plot(avg_spec_moving_flipped, label="Cluster_moving_average_flipped")
        plt.plot(avg_spec_moving, label="Cluster_moving_average")
        plt.plot(avg_spec_moving_clean, label="Cleaned Cluster_moving_average")
        # plt.plot(norm_exp, label="Cluster Average")
        plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.3,0.9), ncol=2)
        plt.xlabel("Time")
        event_fname= "Cluster members="+str(len(X_cluster))+" cluster_"+str(k)
        plt.ylabel("Normalized signal Cluster_"+str(event_fname))
        # plt.title("Final PCC_value from examplar= %s cluster average=%s and moving average =%s" %(pcc_result_exam,pcc_result_avg,pcc_result_mov ))
        plt.title("Final PCC_value from moving average before =%s and after= %s" %(pcc_result_mov, pcc_result_mov_clean))
        plt.grid(False)
        plt.savefig('Results/Sim_Dist_graph_{}.jpg'.format(k))
        plt.tight_layout()
        plt.show() 
        plt.close(fig)           
        for i in range(0,len(X_cluster)):
            r_score=event.pcc(X_cluster[i],X_ex)-event.pcc(X_cluster[i],X_ex[::-1])-2 #let's make between [-2 0] # avg for X_ex X_cluster[i] is the ith cluster the cluster members and instead X_examplar we can use X_cluster average
            r.append(r_score)
        a = np.mean(r); b= np.mean(X_distance);
        rank_score[k]= a/b #np.mean(r)/np.mean(X_distance) 
    temp =(-rank_score).argsort()[:len(rank_score)] #highest score rank 1
    # temp = rank_score.argsort() #lowest score rank 1
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(rank_score))
    ranks_db = DB_rank(X,crit)
    pcc_result_cluster= np.zeros([cluster_count])
    for k in range(0,cluster_count):
        pcc_result_cluster[k]= event.pcc(norm_theory,norm_exp_k[k])
    temp1 = (-pcc_result_cluster).argsort()[:len(pcc_result_cluster)]
    ranks_pcc = np.empty_like(temp1)
    ranks_pcc[temp1] = np.arange(len(pcc_result_cluster))
    fig= plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    plt.rcParams.update({'font.size': 12})
    
    for k in range(0,cluster_count):
        cluster_name= 'cluster_'+str(k)
        cluster_name_all= cluster_name +' True PCC(T,avg_cl)='+str(round(pcc_result_cluster[k],2))+' PCC_rank='+str(ranks_pcc[k]+1) + ' Proposed_rank='+str(ranks[k]+1)
        plt.plot(norm_exp_k[k], label=cluster_name_all)
        plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.3,1.2), ncol=2)
    plt.savefig('Results/All_rank.pdf')
    plt.show()
    plt.tight_layout()
    plt.close(fig)
    return rank_score, pcc_result_cluster, ranks+1, ranks_pcc+1
def sil_cluster(X,ind):
    unique_ind= np.unique(ind)
    new_ind=[]
    silhouette_avg = silhouette_score(X, ind, metric='correlation')
    silhouette_values_samples = silhouette_samples(X, ind, metric='correlation' )
    silhouette_samples_cluster=np.zeros([len(unique_ind)])
    for k in range(0, len(unique_ind)):
        test=[]
        test_all=[]
        for i in range (0, len(ind)):
               if ind[i]==unique_ind[k]:
                   test=silhouette_values_samples[i]
                   test_all.append(test)
               else:
                   print('not matched')
               new_ind.append(k)
        silhouette_samples_cluster[k]=np.mean(test_all)
    temp = -silhouette_samples_cluster.argsort() #lowest score rank 1
    ranks_sil = np.empty_like(temp)
    ranks_sil[temp] = np.arange(len(silhouette_samples_cluster))
    ranks_sil= ranks_sil+1
    return(ranks_sil, silhouette_avg)

def DB_rank(X,crit):
    Examplar = list(crit.keys())
    Examplar = np.array(Examplar)
    cluster_count=len(Examplar)
    S= np.zeros([cluster_count])
    db_score= np.zeros([cluster_count])
    d = np.zeros([cluster_count,cluster_count])
    R= np.zeros([cluster_count,cluster_count])
    for i in range(0,cluster_count):
        cluster_members=list(crit.get(Examplar[i]))
        member_count=len(cluster_members)
        # print(member_count)
        test_all=[]
        for j in range(0, member_count):
            x=X[Examplar[i]]; x= x[::-1]
            test = event.pcc(X[Examplar[i]],X[cluster_members[j]])
            test_all.append(test)
        test_all=np.array(test_all)
        S[i]=np.mean(test_all)
        dist_all=[]
        for j in range(0, cluster_count):
            dist_i_j = event.pcc(X[Examplar[i]],X[Examplar[j]])
            dist_all.append(dist_i_j)
        d[i]=np.array(dist_all)
        for j in range(0, cluster_count):
            R[i,j]= (S[i]+S[j])/d[i,j]
            R[j,i]= R[i,j]
    for i in range(0,cluster_count):
        db_score[i]= max(R[i])
    temp_db = -db_score.argsort() # ascending rank, lowest score rank 0
    ranks = np.arange(len(db_score))[temp_db.argsort()]
    return ranks+1

def get_plot(X,crit,ind,t):
    [Xnew, rnew, label, r_all, r_flip_all]= find_orientation(X,t)
    k=0
    ind=ind
    keysList = list(crit.keys())
    cluster_count=len(keysList)
    ranks_db = DB_rank(X,crit)
    ranks_sil, sil_avg = sil_cluster(X,ind)
    rank_score, pcc_score, ranks_ours, ranks_pcc = get_cluster_rank(X, crit, t)
    for fig_number in range(0,cluster_count):
        x_f=[]
        y_f=[]
        label_f=[]
        for i in range(0, len(X)): 
            if (ind[i]==keysList[k]):# manually label ko value change garnu parxa #95	89	332	457	415 pcc_d#64	343	40	217	195	230	412	85	421	237	447	124	391
                x_f.append(r_all[i])
                y_f.append(r_flip_all[i])
                label_f.append(label[i])
        k=k+1
        ax=plt.axes()
        ax.set(facecolor = "white")
        scatter=plt.scatter(x_f,y_f,c=label_f, cmap='rainbow', alpha=0.7, edgecolors='b') # manually get plot for each cluster
        plt.title("Total spectra in Clusters ="+str(len(label_f))+ "Proposed rank="+str(ranks_ours[fig_number])+" DB_rank="+str(ranks_db[fig_number])+ " Sil_rank= "+str(ranks_sil[fig_number]))    
        plt.xlabel("PCC(S,T)")
        plt.ylabel("PCC(S_flipped,T") 
        plt.legend(*scatter.legend_elements(), loc="upper right", title="Orientation")
        plt.savefig('Results/Sim_PCC_scatter_{}.jpg'.format(fig_number))
        plt.grid(False)
        plt.show()
        plt.close()

def clean_cluster(X_cluster,X_ex, t,count):
    avg = np.mean(X_cluster, axis=0)
    dist=[]
    for i in range(0, len(X_cluster)):
        dist_i = event.pcc(X_ex,X_cluster[i])+1 #[0 2]
        dist.append(dist_i)
    dist = np.array(dist)
    temp = (-dist).argsort()[:len(dist)] # highest score rank 0
    rank = np.empty_like(temp)
    rank[temp] = np.arange(len(dist))
    threshold= round(0.8*len(rank))
    X_cluster_clean= np.zeros([threshold,X_cluster.shape[1]])
    r1=[]
    r_flip1=[]
    label_f=[]
    for i in range(0,threshold):
        k = rank[i]
        print('k=',k)
        x = X_cluster[k]
        X_cluster_clean[i]=x
        x_flip=x[::-1]
        x = np.reshape(x, t.shape)
        r = np.corrcoef(x, t)[0,1]
        r1.append(r)
        r_flip= np.corrcoef(x_flip,t)[0,1]
        r_flip1.append(r_flip)
        if r>r_flip:
            label=1
        else:
            label=0
        label_f.append(label)
    ax=plt.axes()
    ax.set(facecolor = "white")
    plt.rcParams.update({'font.size': 12})
    # plt.subplot(2,1,2)
    scatter=plt.scatter(r1,r_flip1,c=label_f, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.title("Total spectra in clean Clusters ="+str(len(label_f)))    
    plt.xlabel("PCC(S,T)")
    plt.ylabel("PCC(S_flipped,T)") 
    plt.legend(*scatter.legend_elements(), loc="upper right", title="Orientation")
    plt.savefig('Results/Sim_clean_scatter_{}.jpg'.format(count))
    plt.grid(False)
    plt.show()
    plt.close()
    return X_cluster_clean          