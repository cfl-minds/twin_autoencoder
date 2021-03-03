import autograd.numpy as np
from autograd import grad, jacobian
#from autograd.scipy.stats import norm
from scipy.optimize import minimize
#import statsmodels.api as sm
from pandas import read_csv as read
import matplotlib.pyplot as plt

#####import data
train = read('Depth5/mse_train.csv')
valid = read('Depth5/mse_valid.csv')
test  = read('Depth5/mse_test.csv')

train.columns = ['index', 'e_s', 'e_f']
valid.columns = ['index', 'e_s', 'e_f']
test.columns  = ['index', 'e_s', 'e_f']

#print(train.head())

def NLL(theta):
    
    a = theta[0]
    b = theta[1]
    c = theta[2]
    
    mu    = a * e_s + b
    sigma = c * e_s
    
    ll = -N/2 * np.log(2*np.pi) - np.sum(np.log(sigma)) - 0.5 * np.sum(((e_f-mu)/ sigma)**2)
    return -ll/N
    
def NLL_valid(theta):
    
    a = theta[0]
    b = theta[1]
    c = theta[2]
    
    mu    = a * e_s_v + b
    sigma = c * e_s_v
    
    ll = - N_v/2 * np.log(2*np.pi) - np.sum(np.log(sigma)) - 0.5 * np.sum(((e_f_v-mu)/ sigma)**2)
    return -ll/N_v

tol  = 5000000
tol2 = 5000000

e_s  = np.asarray(train.e_s[ (train.e_f < tol) & (train.e_s < tol2) ])
e_f  = np.asarray(train.e_f[ (train.e_f < tol) & (train.e_s < tol2) ])
N    = len(e_s)
e_s_v  = np.asarray(valid.e_s[ (valid.e_f < tol) & ( valid.e_s < tol2) ])
e_f_v  = np.asarray(valid.e_f[ (valid.e_f < tol) & ( valid.e_s < tol2) ])
N_v    = len(e_s_v)

print(N, N_v)

def callback(x):
    train_loss = NLL(x)
    valid_loss = NLL_valid(x)
    history.append(np.asarray([x, train_loss, valid_loss]))

jacobian_  = jacobian(NLL)
theta_start = np.array([0.1, 0, 0.1])
history = []
res1 = minimize(NLL, theta_start, method = 'BFGS', options={'disp': False, 'maxiter': 200}, jac=jacobian_, callback=callback)
history = np.reshape(history, (res1.nit, 3))
print(history)
print("Convergence Achieved: ", res1.success)
print("Number of Function Evaluations: ", res1.nfev)


fig, ax = plt.subplots()
ax = plt.axes([0,0,1,1])

dq = np.hstack((np.arange(history.shape[0]).reshape((history.shape[0],1)), history[:,1].reshape((history.shape[0],1)), history[:,2].reshape((history.shape[0],1))))
#np.savetxt('fig13.csv', dq, delimiter=',', header='iteration,train,valid')

plt.plot(np.arange(history.shape[0]), history[:,1], label='train')
plt.plot(np.arange(history.shape[0]), history[:,2], label='valid')
#plt.yscale('log')
plt.grid()
#plt.xlim(0,6e-4)
#plt.ylim(0,1.5e-4)
plt.xlabel('iteration')
plt.ylabel('NLL')
plt.legend(loc='upper right')

plt.savefig('images/nnl_iter.png', dpi=300, bbox_inches='tight',pad_inches = 0)
plt.close()
########
index = np.argmin(history[:,2])#history.shape[0]-1#
a,b,c = history[index,0]
print('Calibration finished, the values of (a,b,c) is ', a,b,c, ' !')




################## The scatter plot 
es        = np.arange(0, 2e-4, 1e-5)#np.array([0,4e-5,8e-5,1.2e-4,1.6e-4, 2e-4])
mu        = a * es + b
one_sigma = c * es

fig, ax = plt.subplots()
ax = plt.axes([0,0,1,1])
plt.plot(test.e_s, test.e_f, 's', color = 'green', markersize=2)
plt.plot(es, mu, color='black', linewidth=5)
ax.fill_between(es, mu - 1.0 * one_sigma, mu + 1.0 * one_sigma, facecolor='gray', alpha=0.5)
plt.grid()
plt.xlim(0,2e-4)
plt.ylim(0,1.1e-4)
plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
plt.xlabel('MSE_reconstruction')
plt.ylabel('MSE_flow')
plt.savefig('/images/err_interval.png', dpi=300, bbox_inches='tight',pad_inches = 0)
plt.close()






























