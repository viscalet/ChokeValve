# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:52:47 2022

Contain full functionalities of model f
Ready to be wrapped in a toolbox

@author: xinghenl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

from matplotlib import cm
from sklearn.ensemble import IsolationForest
from statsmodels.nonparametric.kernel_regression import KernelReg


#%% Read data

'''
User input:
    well_id can be p1w1, p1w2...p1w7 or p3w1, p3w2, p3w3
    Cv_file contains the theoretical Cv 
'''

well_id = 'p1w2'

data_equinor = pd.read_csv(
    f'C:\\Users\\xinghenl\\Downloads\\EquinorData\\NewData\\{well_id}.csv',
    header=0)

# Cv: Cv428, Cv110
Cv_file = pd.read_csv(
    'C:\\Users\\xinghenl\\Downloads\\EquinorData\\NewData\\Cv428.csv',
    header=0)

Cv_file.columns = ['Column1', 'ChokeCv', 'Travel']
data_equinor.columns = ['Timestamp',
                        'PercentTravel',
                        'PressureDrop',
                        'ELF',
                        'Flow',
                        'Sachdeva'
                        ]

#%% Regression for baseline Cv data (logistic and polynomial)

'''
Interpolation of the theoretical Cv
'''

axis_h = np.linspace(0, 100, 200)

kde_cv = KernelReg(endog=Cv_file['ChokeCv'], 
                exog=Cv_file['Travel'], 
                var_type='c',
                bw=[3])

cv_pred = kde_cv.fit(axis_h)


#%% Visualization: Cv data points and the logistic regression.

# Cv data points

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(Cv_file['Travel'], Cv_file['ChokeCv'],alpha=0.5)
ax.plot(axis_h, cv_pred[0], '-', color='tab:blue', alpha=0.8)

plt.show()


# Visualization: Cv data points and the logistic regression.
plt.plot(axis_h,
         kde_cv.fit(axis_h)[0],
         label='kernel regression'
         )
plt.scatter(Cv_file['Travel'],
            Cv_file['ChokeCv'],
            label='Cv data points',
            c='red'
            )
plt.xlabel('Choke travel %', fontsize=15)
plt.ylabel('Theoretical Cv', fontsize=15)
plt.legend(fontsize=15)


# Erosion overview: theoretical Cv and observed Cv (ELF).
plt.scatter(data_equinor['PercentTravel'],
            kde_cv.fit(data_equinor['PercentTravel'])[0],
            label='Theoretical Cv',
            )

plt.scatter(data_equinor['PercentTravel'],
            data_equinor['Sachdeva'],
            label='Sachdeva', marker='*',
            )
plt.xlabel('Percent travel', fontsize=30)
plt.ylabel('Cv', fontsize=30)
plt.legend(fontsize=30)

plt.scatter(data_equinor['PercentTravel'],
            data_equinor['ELF'],
            label='ELF',
            )

#%% Preprocesssing: remove nan values and repeated timestamp

# data_to_use can be Sachdeva or ELF

def data_preprocess(data, data_to_use):
    data_new = data.drop([0],axis=0) if data[data_to_use][0]==0 else data
    data_new = data_new[data_new[data_to_use].notna()]

    # Compute new delta Cv using polynomial fit
    v = kde_cv.fit(data_new['PercentTravel'])[0]
    data_4 = data_new.assign(baseline = v)

    # Add delta Cv to the dataframe
    data_4['deltaCv'] = data_4[data_to_use] - data_4['baseline'] 

    # The timestamps are converted to operational days.
    Days4 = (pd.to_datetime(data_4['Timestamp'], dayfirst=True)
            - pd.to_datetime(data_4['Timestamp'], dayfirst=True)[data_new.first_valid_index()]
            ).dt.days

    data_4['days'] = Days4

    # Check duplicates of timestamp

    days_duplicates = (
        [item for item, 
         count in collections.Counter(Days4).items() if count > 1])
    idx_todrop = np.empty(0)
    for d in days_duplicates:
        idxs = (data_4.index[data_4['days']==d])
        idx_todrop = np.append(idx_todrop, idxs[1:])

    data_4 = data_4.drop(idx_todrop,axis=0)
    
    return data_4

data_new = data_preprocess(data_equinor, 'Sachdeva')


#%% Outlier detection

# Useless features: Timestamp, days
# Other features: ELF or Sachdeva, baseline, deltaCv
# columns_to_drop = ['Timestamp','days','ELF','baseline', 'deltaCv']

columns_to_drop = ['Timestamp','days']

def outlier_score(data, columns_to_drop):
    data_partial = data.drop(columns_to_drop, axis=1)
    clf_IF = IsolationForest(random_state=0,
                             contamination='auto').fit(data_partial)
    pr_IF = clf_IF.predict(data_partial)
    #scores_IF = clf_IF.score_samples(data_partial)
    return pr_IF
    

data_new['score'] = outlier_score(data_new, columns_to_drop)


plt.scatter(data_new['PercentTravel'],
            data_new['deltaCv'],
            c = data_new['score'],
            label='Outlier scores',
            cmap='plasma')
plt.xlabel('Percent travel', fontsize=30)
plt.ylabel('delta Cv', fontsize=30)
plt.colorbar()


plt.scatter(data_new['days'],
            data_new['deltaCv'],
            c = data_new['score'],
            label='Outlier scores',
            cmap='plasma')
plt.xlabel('days', fontsize=30)
plt.ylabel('delta Cv', fontsize=30)
plt.colorbar()


plt.scatter(data_new['days'],
            data_new['PercentTravel'],
            c = data_new['score'],
            label='Outlier scores',
            cmap='plasma')
plt.xlabel('days')
plt.ylabel('PercentTravel')
plt.colorbar()


#%% determine the dataset for further analysis


if well_id == 'p1w1':
    t1 = 150
    t2 = 300
    data_n = data_new[((data_new["days"]<=t1) | (data_new["days"]>=t2))
                      & (data_new["PercentTravel"]>10)]
elif well_id =='p1w2':
    t1 = 170
    t2 = 300
    data_n = data_new[((data_new["days"]<=t1) | (data_new["days"]>=t2))
                      & (data_new["PercentTravel"]>10)]
elif well_id=='p1w3':
    data_n = data_new[data_new['score']==1]
elif well_id=='p1w6':
    t1 = 150
    t2 = 250
    data_n = data_new[((data_new["days"]<=t1) | (data_new["days"]>=t2))
                      & (data_new["score"]==1)]
    
    

#%% Visualization Raw Cv, Reference Cv and delta Cv
    
plt.plot(data_n['days'],
         data_n['deltaCv'],
         label='Cv difference'
         )
plt.plot(data_n['days'],
         data_n['Sachdeva'],
         label='Raw Cv'
         )
plt.plot(data_n['days'],
         kde_cv.fit(data_n['PercentTravel'])[0],
         label='Reference'
         )
plt.xlabel('Relative time (days)', fontsize=15)
plt.ylabel('Cv', fontsize=15)
plt.legend(fontsize=12)

plt.plot(data_n['days'],
         data_n['PercentTravel'],
         label='Opening'
         )
plt.xlabel('Relative time (days)', fontsize=15)
plt.ylabel('Relative Opening', fontsize=15)
plt.legend(fontsize=15)

plt.scatter(
         data_n['PercentTravel'],
         data_n['deltaCv'],
         label='Opening'
         )
plt.xlabel('Opening', fontsize=15)
plt.ylabel('delta Cv', fontsize=15)
plt.legend(fontsize=15)

#%% Construct finite difference data set containing diff travel and diff deltaCv

data_diff = pd.DataFrame(
    np.transpose([np.diff(data_n['deltaCv']),
                  np.diff(data_n['PercentTravel']),
                  np.diff(data_n['days']),
                  np.asarray(data_n['PercentTravel'])[0:-1],
                  np.asarray(data_n['deltaCv'])[0:-1],
                  np.asarray(data_n['days'])[0:-1],
                  ]),
    columns=['deltaX',
             'deltaH',
             'gap',
             'actual_h',
             'actual_X',
             'days',
             ])

#%% Visualization: scatter matrix of data_diff, delta h versus delta X

plt.scatter(data_diff['deltaH'],
            data_diff['deltaX'])
plt.xlim([-2,2])
plt.xlabel('delta h', fontsize=20)
plt.ylabel('delta X',fontsize=20)


#%% Vectorized visualization of Cv measurements


plt.quiver(data_diff['actual_h'], 
           data_diff['actual_X'], 
           data_diff['deltaH'], 
           data_diff['deltaX'],
           data_diff['days'],
           width=0.003,
           headwidth = 10,
           headlength = 7,
           angles='xy', scale_units='xy', scale=1)
plt.xlabel('opening', fontsize = 15)
plt.ylabel('x', fontsize = 15)
plt.colorbar()


plt.quiver(np.zeros([1,len(data_diff)]), 
           np.zeros([1,len(data_diff)]),
           data_diff['deltaH'], 
           data_diff['deltaX'],
           data_diff['days'],
           width=0.003,
           angles='xy', scale_units='xy', scale=1)
plt.xlabel('∆h', fontsize = 40)
plt.ylabel('∆x', fontsize = 40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(np.min(data_diff['deltaH']), 
         -np.min(data_diff['deltaH']))
plt.ylim(np.min(data_diff['deltaX']), 
         -np.min(data_diff['deltaX']))
plt.colorbar()




#%% function: get parameters

def get_params(data, p, q):
    vec_t = np.array(data['days'])
    vec_h = np.array(data['PercentTravel'])
    vec_dt = np.diff(data['days'])
    vec_dx = np.diff(data['deltaCv'])
    
    # p: polynomial order for the initial shape
    n = len(vec_dt)
    hp = []
    for i in range(1,p+1):
        hp.append(np.diff(vec_h**i))
    Hp = np.reshape(hp,[p,n])
    HP = Hp.T

    # q: polinomial order for calibration function g
    
    if q==0:
        mat_d = HP
    else:
        hq = []
        for i in range(0,q+1):
            hq.append(vec_dt*(vec_h[1:])**i+vec_t[0:-1]*(np.diff(vec_h**i)))
        Hq = np.reshape(hq,[q+1,n])
        HQ = Hq.T
        mat_d = np.concatenate((HP, HQ), axis=1)

    param_f = np.matmul(np.linalg.inv(np.matmul(mat_d.transpose(),
                                                mat_d)), 
                        np.matmul(mat_d.transpose(),vec_dx))

    return param_f
    

def get_f(param_p, vec_openings, data):
    p = len(param_p)
    vec_h = np.array(data['PercentTravel'])
    mat_f0 = []
    for i in range(1,p+1):
        mat_f0.append(vec_openings**i-vec_h[0]**i)
    mat_f0 = np.reshape(mat_f0,[p,len(vec_openings)])
    f0_h = np.array(data['deltaCv'])[0]+np.matmul(param_p,mat_f0)
    return f0_h
    
def get_g(param_q, vec_openings):
    q = len(param_q)
    mat_c0 = []
    for i in range(0,q):
        mat_c0.append(vec_openings**i)
    mat_c0 = np.reshape(mat_c0,[q,len(vec_openings)])
        
    c0 = np.matmul(param_q,mat_c0)
    return c0
    
def get_incres(data, param_p, param_q):
    vec_t = np.array(data['days'])
    vec_h = np.array(data['PercentTravel'])
    vec_dt = np.diff(data['days'])
    vec_dx = np.diff(data['deltaCv'])
    
    # p: polynomial order for the initial shape
    n = len(vec_dt)
    p = len(param_p)
    hp = []
    for i in range(1,p+1):
        hp.append(np.diff(vec_h**i))
    Hp = np.reshape(hp,[p,n])
    HP = Hp.T

    # q: polinomial order for calibration function g
    q = len(param_q)
    hq = []
    for i in range(0,q):
        hq.append(vec_dt*(vec_h[1:])**i+vec_t[0:-1]*(np.diff(vec_h**i)))
    Hq = np.reshape(hq,[q,n])
    HQ = Hq.T
    mat_d = np.concatenate((HP, HQ), axis=1)
    param_f = np.concatenate((param_p,param_q))
    incres = vec_dx - np.matmul(mat_d, param_f)
    return incres
    
#%% Cross validation

nb_cv = 3000
ratio = 0.75
model_config = [[1,0],[1,1],
                [2,0],[2,1],
                [3,0],[3,1]]

def cv_perm(nb_cv, ratio, model_config, data):
    residuals = []
    
    for i in range(nb_cv):
        idx_add = np.random.permutation(data.index)[:int(ratio*len(data))]
        train_i = data.loc[np.sort(idx_add)] 
        
        for config in (model_config):
            p, q = config[0], config[1]
            param_A = get_params(train_i, p, q)
            param_pA, param_qA = param_A[0:p], param_A[p:]
            incres_A = get_incres(data, param_pA, param_qA)
            residuals.append(np.inner(incres_A, incres_A))
    
    res = np.reshape(residuals,[nb_cv,len(model_config)])
    mean_err = np.mean(res, axis=0)
    return mean_err

mean_er = cv_perm(nb_cv, ratio, model_config, data_n)

best_config = model_config[np.argmin(mean_er)]
p_best, q_best = best_config[0], best_config[1]

#%% Best model: parameter estimation and confidence interval
# vector of openings and initial Cv curve
vec_openings = np.linspace(np.min(data_n['PercentTravel']),
                           np.max(data_n['PercentTravel']),
                           100)


param = get_params(data_n, p_best, q_best)
param_p, param_q = param[0:p_best], param[p_best:]
incres = get_incres(data_n, param_p, param_q)

f0 = get_f(param_p, vec_openings, data_n)

plt.plot(vec_openings, f0, color='red', label='f')
plt.quiver(data_diff['actual_h'], 
           data_diff['actual_X'], 
           data_diff['deltaH'], 
           data_diff['deltaX'],
           data_diff['days'],
           width=0.003,
           angles='xy', scale_units='xy', scale=1)
plt.xlabel('opening', fontsize = 15)
plt.ylabel('x', fontsize = 15)
plt.colorbar()

g0 = get_g(param_q, vec_openings)

plt.plot(vec_openings, g0, label='model')
plt.xlabel("valve opening", fontsize=20)
plt.ylabel("average daily increment", fontsize=20)



#%% Get erosion surface

vec_t = np.array(data_n['days'])
vec_h = np.array(data_n['PercentTravel'])
vec_dt = np.diff(data_n['days'])
vec_dx = np.diff(data_n['deltaCv'])
vec_dh = np.diff(data_n['PercentTravel'])
n = len(vec_dt)

Curve = [f0]
for i in range(n):
    new_curve = Curve[i]+incres[i]+vec_dt[i]*g0
    Curve.append(new_curve)
    
Curve = np.array(Curve)

X_time, Y_open = np.meshgrid(vec_t, vec_openings)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf_curve = ax.plot_surface(X_time,
                                Y_open,
                                Curve.T,
                cmap=cm.coolwarm)
ax.set_xlabel('Time in days',fontsize=20)
ax.set_ylabel('Valve opening',fontsize=20)
ax.set_zlabel('Cv deviation',fontsize=20)
fig.colorbar(surf_curve)


#%% Smoothed surface

# Fixed bandwidth
bandwidth = 30
time_pred = np.arange(vec_t[0],vec_t[-1]+20,1)

smoothed_curve = []
for i in range(len(vec_openings)):
    datai = Curve[:,i]
    kde_i = KernelReg(endog=datai, 
                    exog=vec_t, 
                    var_type='c',
                    bw=[bandwidth])
#                    bw='cv_ls')

    estimator_i = kde_i.fit(time_pred)
    smoothed_curve.append(estimator_i[0])

smoothed_curve = np.array(smoothed_curve).transpose()
X_times, Y_opens = np.meshgrid(time_pred, vec_openings)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(data_n['days'],
           data_n['PercentTravel'],
           data_n['deltaCv'])
surf_smooth = ax.plot_surface(X_times,
                              Y_opens,
                              smoothed_curve.T,
                cmap=cm.coolwarm)
ax.set_xlabel('Time in days',fontsize=20)
ax.set_ylabel('Valve opening',fontsize=20)
ax.set_zlabel('Cv deviation',fontsize=20)
fig.colorbar(surf_smooth)

#%% Parameter uncertainty


def get_X(data, param_p, param_q):
    vec_t = np.array(data['days'])
    vec_h = np.array(data['PercentTravel'])
    vec_dt = np.diff(data['days'])
    
    # p: polynomial order for the initial shape
    n = len(vec_dt)
    p = len(param_p)
    hp = []
    for i in range(1,p+1):
        hp.append(np.diff(vec_h**i))
    Hp = np.reshape(hp,[p,n])
    HP = Hp.T

    # q: polinomial order for calibration function g
    q = len(param_q)
    hq = []
    for i in range(0,q):
        hq.append(vec_dt*(vec_h[1:])**i+vec_t[0:-1]*(np.diff(vec_h**i)))
    Hq = np.reshape(hq,[q,n])
    HQ = Hq.T
    mat_d = np.concatenate((HP, HQ), axis=1)
    return mat_d

Mat_X = get_X(data_n, param_p, param_q)
Mat_C = np.linalg.inv(np.matmul(Mat_X.transpose(), Mat_X))
sg_hat = np.inner(incres,incres)/(len(data_n)-len(param))
var_param = sg_hat*np.diag(Mat_C)
sqrt_param = np.sqrt(var_param)


#%% Corrlated parameter uncertainty analysis

from scipy.stats import multivariate_normal


def resample_f(config, data, nb_sample, vec_openings):
    
    vec_t = np.array(data['days'])
    vec_dt = np.diff(data['days'])
    n = data.shape[0]
    
    p, q = config[0], config[1]
    param = get_params(data, p, q)
    param_p, param_q = param[0:p], param[p:]
    
    Mat_X = get_X(data, param_p, param_q)
    Mat_C = np.linalg.inv(np.matmul(Mat_X.transpose(), Mat_X))
    sg_hat = np.inner(incres,incres)/(len(data)-len(param))
    covx = sg_hat*Mat_C
    
    
    list_curve = []
    list_s,list_h,list_t,list_d = [],[],[],[]

    for i in range(nb_sample):
        
        param_i = multivariate_normal.rvs(mean=param, 
                                          cov=covx)
        
        param_pi = param_i[0:p]
        if len(param_pi)==1:
            param_pi = np.asarray(param_pi).reshape(-1)
        
        param_qi = param_i[p:]
        if len(param_qi)==1:
            param_qi = np.asarray(param_qi).reshape(-1)
        
        incres_i = get_incres(data, param_pi, param_qi)
    
        fi = get_f(param_pi, vec_openings, data)
        gi = get_g(param_qi, vec_openings)
        
        for k in range(len(vec_openings)):
            list_s.append(i)
            list_h.append(vec_openings[k])
            list_t.append(vec_t[0])
            list_d.append(fi[k])
            
        Curve_i = [fi]
        for j in range(1,n):
            new_curve = Curve_i[j-1]+incres_i[j-1]+vec_dt[j-1]*gi
            Curve_i.append(new_curve)
            for k in range(len(vec_openings)):
                list_s.append(i)
                list_h.append(vec_openings[k])
                list_t.append(vec_t[j])
                list_d.append(new_curve[k])
                
                     
        Curve_i = np.array(Curve_i)
        list_curve.append(Curve_i)
    
    dfcv = pd.DataFrame({'N_sample':list_s,
                         'Opening':list_h, 
                         'Time':list_t,
                         'deltaCv':list_d})
    
    curves = np.array(list_curve)
    return curves, dfcv

vec_openings = np.linspace(np.min(data_n['PercentTravel']),
                           np.max(data_n['PercentTravel']),
                           50)
nb_sample = 1000
Curves, Df2 = resample_f(best_config, data_n, nb_sample, vec_openings)


#%% joy plot for a single opening

import joypy

h_idx = 45    
df_50 = Df2[Df2['Opening']==vec_openings[h_idx]]

labels=[y if y%10==0 else None for y in list(df_50.Time.unique())]
joypy.joyplot(df_50, by="Time", labels=labels, column="deltaCv",
              grid="y",fade=True, colormap=cm.viridis)


#%%  quarters

h_idx = 45
vec_openings[h_idx]

vec_ub = []
vec_lb = []
for i in range(len(vec_t)):
    data_i = Curves[:,i,h_idx]
    vec_ub.append(np.quantile(data_i, 0.95))
    vec_lb.append(np.quantile(data_i, 0.05))
    

plt.scatter(vec_t, data_n['deltaCv'],
         label='Raw Cv deviation', color='orange', s=15)
plt.fill_between(vec_t, vec_ub, vec_lb, alpha=0.5, color='blue')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Cv deviation', fontsize=15)
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.legend(loc='lower right', fontsize=15)
