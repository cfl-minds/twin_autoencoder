import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


e_f_star = 5e-5# give a tolerance on the mean squared error of flow prediction


### load MSE on the training/validation/test set
mse_tr = pd.read_csv('Depth5/mse_train.csv')
mse_tr.columns = ['index', 'shapes', 'flow']

mse_va = pd.read_csv('Depth5/mse_valid.csv')
mse_va.columns = ['index', 'shapes', 'flow']

mse_te = pd.read_csv('Depth5/mse_test.csv')
mse_te.columns = ['index', 'shapes', 'flow']

### search the optimal threshold e_s_star given that tolerance e_f_star
f_tr = list()
f_va = list()

interval = np.arange(1e-7, 5e-4, 1e-8)
for e_s in interval:
    
    FN = (( mse_tr.shapes < e_s ) & ( mse_tr.flow > e_f_star)).sum()
    FP = (( mse_tr.shapes > e_s ) & ( mse_tr.flow < e_f_star)).sum()
    f_tr.append( (FN + FP) / len(mse_tr) )
    
    FN = (( mse_va.shapes < e_s ) & ( mse_va.flow > e_f_star)).sum()
    FP = (( mse_va.shapes > e_s ) & ( mse_va.flow < e_f_star)).sum()
    f_va.append( (FN + FP) / len(mse_va) )

fig = plt.figure()
ax = plt.axes([0,0,1,1])
plt.plot(interval, f_tr, label='train')
plt.plot(interval, f_va, label='valid')
#plt.axis('off')
plt.grid()
plt.xlabel('e_s')
plt.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title('Mistaken classification rate')
plt.legend(loc='upper right')
plt.savefig('images/mistake_rate.png', dpi=300, bbox_inches='tight',pad_inches = 0)


e_s_star = interval[np.argmin(f_va)]
print('Given the tolerance e_f* = {}, '.format(e_f_star), 'the optimal threshold is {}. '.format(e_s_star), 'Mistake rate on the validation set is {}. '.format(f_va[np.argmin(f_va)]))
##################################



### try multiple tolerance e_f_star, and see how the optimal threshold e_s_star changes
results = dict()
for e_f_star in np.arange(1e-5, 1e-4, 1e-5):
    f_tr = list()
    f_va = list()
    interval = np.arange(1e-6, 4e-4, 1e-7)
    for e_s in interval:
    
        FN = (( mse_tr.shapes < e_s ) & ( mse_tr.flow > e_f_star)).sum()
        FP = (( mse_tr.shapes > e_s ) & ( mse_tr.flow < e_f_star)).sum()
        f_tr.append(( FN + FP ) / len(mse_tr))
    
        FN = (( mse_va.shapes < e_s ) & ( mse_va.flow > e_f_star)).sum()
        FP = (( mse_va.shapes > e_s ) & ( mse_va.flow < e_f_star)).sum()
        f_va.append(( FN + FP ) / len(mse_va))


    index = np.argmin(f_va)
    e_s_star = interval[index]

    FN = (( mse_te.shapes < e_s_star ) & ( mse_te.flow > e_f_star)).sum()
    FP = (( mse_te.shapes > e_s_star ) & ( mse_te.flow < e_f_star)).sum()

    results[e_f_star] = np.array([e_s_star, f_tr[index], f_va[index],(FN + FP)/len(mse_te)])


all_e_f = np.arange(1e-5, 1e-4, 1e-5)
all_e_s = np.array(list(results.values()))[:,0]

fig = plt.figure()
ax = plt.axes([0,0,1,1])
plt.plot(all_e_f, all_e_s)
plt.ticklabel_format(axis="both", style="scientific", scilimits=(0,0))
plt.grid()
plt.xlabel('e_f_star')
plt.title('e_s_star as a function of e_f_star')
plt.savefig('images/threshold_versus_tolerance.png', dpi=300, bbox_inches='tight',pad_inches = 0)


### see how the mistake rate changes versus the tolerance e_f_star
fig = plt.figure()
ax = plt.axes([0,0,1,1])
plt.plot(all_e_f, np.array(list(results.values()))[:,1], label='train')
plt.plot(all_e_f, np.array(list(results.values()))[:,2], label='validation')
plt.plot(all_e_f, np.array(list(results.values()))[:,3], label='test')
plt.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('e_f_star')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title('False classification rate as a function of e_f_star')
plt.savefig('images/Mistake_rate_versus_tolerance.png', dpi=300, bbox_inches='tight',pad_inches = 0)

