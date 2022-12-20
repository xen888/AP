import pandas
import numpy as np
import csv
import sys
from Events import event


theory_fname = '202212_nanospectra_code/oxa_181/lt119/theory_trace_2000.csv' #sys.argv[1]
event_fname = 'Results/Oxa_181/PCC_D/cluster13.csv' #sys.argv[2]
plot_fname = 'test.pdf' #sys.argv[3]
print("theory file name", theory_fname)
aa_fname= 'nanospectra data/AB42.csv'
  
df = pandas.read_csv(theory_fname, header=None)
theory_data = df.to_numpy()
#print(theory_data)
theory_spec = event.Event(0, theory_data[0,:])
#print(theory_data)
df = pandas.read_csv(event_fname, header=None)
exp_data = df.to_numpy()

avg = np.mean(exp_data, axis=0)
avg_spec = event.Event(0, avg)

df = pandas.read_csv(aa_fname, header=None)
aa_label = df


norm_spec = event.normalize(avg_spec.data)
norm_theory = event.normalize(theory_spec.data)
avg_spec_moving= event.moving_averages(norm_spec, 24) #moving_averages(avg_spec.data, 24) #also defined in event but not working
norm_theory_moving= event.moving_averages(norm_theory,24)
# print("dist", dist)
pcc_result=event.pcc(norm_theory, norm_spec)
pcc_result=str(round(pcc_result, 3))
print("PCC between ref and avg with Normalised average=",pcc_result)
pcc_result_mov=event.pcc(norm_theory_moving, avg_spec_moving)
pcc_result_mov=str(round(pcc_result_mov, 3))
print("PCC between ref and avg with Normalised moving average=",pcc_result_mov)

# pcc_result_mov=event.pcc(norm_align_theory,avg_spec_moving)
# pcc_result_mov=str(round(pcc_result_mov, 3))
# print("PCC between moving ref and avg", pcc_result_mov)
# modified here
#%%
from matplotlib import pyplot as plt
fig= plt.figure()
fig.set_figwidth(20)
fig.set_figheight(10)
ax=plt.axes()
ax.set(facecolor = "white")
plt.rcParams.update({'font.size': 12})
# test=aa_label[0]
# test=test[0]
# x_vals = range(0, len(align_theory))
# aa = np.array(list(test))
# interval_len = (len(align_theory)/len(aa))
# x_tick_vals = np.zeros(len(aa))
# for i in range(0, len(aa)):
#       x_tick_vals[i] = (i*interval_len)
# ax1 = fig.add_subplot(111)
plt.grid(False)
plt.plot(norm_theory_moving, label="Normalised moving Theoretical")
plt.plot(avg_spec_moving, label="Normalised moving Experimental")
# plt.plot(norm_theory, label="Normalised Theory")
# plt.plot(norm_spec, label="Normalised Experimental")
# avg_spec_moving= np.array(avg_spec_moving)
# ax1.plot(avg_spec_moving, label="Experimental_moving")
# plt.plot(avg_spec.data, label="Experimental")
plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.3,1.2), ncol=2)
plt.xlabel("Time")
plt.ylabel("Normalized signal "+str(event_fname))
# plt.title("Final PCC_value from average consensus ="+str(pcc_result))
plt.title("Final PCC_value from consensus average=%s" %(pcc_result_mov))
# ax2 = ax1.twiny()
# plt.xticks(x_tick_vals,aa,fontsize=8)
# plt.tight_layout()
# ax2.set_xlabel('Putative AA position')
plt.show()
# plt.savefig(plot_fname)