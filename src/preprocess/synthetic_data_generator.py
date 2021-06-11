# This synthetic data generator bases on Omi's FullyNN project.
# The original code formed as a jupyter notebook can be retrieved from https://github.com/omitakahiro/NeuralNetworkPointProcess.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.stats import lognorm,gamma
from scipy.optimize import brentq

np.random.seed(12345)

######################################################
### homogeneous possion process
######################################################
def generate_stationary_poisson():
    tau = np.random.exponential(size=64)
    T = tau.cumsum()
    score = np.ones_like(T)
    return T, score

######################################################
### hawkes process
######################################################
def generate_hawkes1():
    [T,LL] = simulate_hawkes(128,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL
    return T, score

def generate_hawkes2():
    [T,LL] = simulate_hawkes(128,0.2,[0.4,0.4],[1.0,20.0])
    score = - LL
    return T, score

def simulate_hawkes(n,mu,alpha,beta):
    T = []
    LL = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            
            if count == n:
                break
        
    return [np.array(T),np.array(LL)]

######################################################
### stationary renewal process
######################################################
def generate_stationary_renewal():
    s = np.sqrt(np.log(6*6+1))
    mu = -s*s/2
    tau = lognorm.rvs(s=s,scale=np.exp(mu),size=64)
    lpdf = - lognorm.logpdf(tau,s=s,scale=np.exp(mu))
    T = tau.cumsum()
    
    return T, lpdf
 
######################################################
### self-correcting process
######################################################
def generate_self_correcting():
    
    def self_correcting_process(mu,alpha,n):
    
        t = 0; x = 0;
        T = [];
        log_l = [];
        Int_l = [];
    
        for i in range(n):
            e = np.random.exponential()
            tau = np.log( e*mu/np.exp(x) + 1 )/mu # e = ( np.exp(mu*tau)- 1 )*np.exp(x) /mu
            t = t+tau
            T.append(t)
            x = x + mu*tau
            log_l.append(x)
            Int_l.append(e)
            x = x -alpha

        return [np.array(T),np.array(log_l),np.array(Int_l)]
    
    [T,log_l,Int_l] = self_correcting_process(1,1,64)
    score = -(log_l - Int_l)
    
    return T, score

def simulate_hawkes(n,mu,alpha,beta):
    T = []
    LL = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            
            if count == n:
                break
        
    return [np.array(T),np.array(LL)]

def simulate_multi_hawkes(n,mu,alpha,beta,head):
    T = head.tolist()
    LL = []

    event_gap = head[1] - head[0]
    x = head[1]
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*(step+event_gap)) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*(step+event_gap))
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            event_gap = T[-1] - T[-2]
            
            if count == n:
                break
        
    return [np.array(T[-n:]),np.array(LL)]

data = {'index': [], 'time_seq': [], 'score': []}
file = 'train'
for i in range(2600):
    time, score = generate_hawkes1()
    data['index'].append(i)
    data['time_seq'].append(time.tolist())
    data['score'].append(score.tolist())

final = pd.DataFrame.from_dict(data)
final.to_json(file + '.json')

data = {'index': [], 'time_seq': [], 'score': []}
file = 'test'
for i in range(260):
    time, score = generate_hawkes1()
    data['index'].append(i)
    data['time_seq'].append(time.tolist())
    data['score'].append(score.tolist())

final = pd.DataFrame.from_dict(data)
final.to_json(file + '.json')
