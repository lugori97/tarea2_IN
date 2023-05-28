# My Utility : auxiliars functions

import pandas as pd
import numpy  as np


def load_config():
    csv_sae = np.loadtxt('cnf_sae.csv',dtype=float)
    csv_sft = np.loadtxt('cnf_softmax.csv',dtype=float) 
    
    cnf_sae = {
        'p_inverse' : csv_sae[0],
        'act_function_encoder': csv_sae[1],
        'sae_max_iter': csv_sae[2],
        'sae_miniBatch_size': csv_sae[3],
        'sae_learning_rate':csv_sae[4],
        'encoder_nodes': csv_sae[5:]
    }

    cnf_sft = {
        'sft_max_iter': csv_sft[0],
        'sft_learning_rate': csv_sft[1],
        'sft_miniBatch_size': csv_sft[2],
    }

    return cnf_sae,cnf_sft

# Initialize one-wieght    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
    
# # STEP 1: Feed-forward of AE
def ae_forward(x, w1, w2, act_func):
    A = [x]
    Z = [np.dot(w1, A[0])]
    A.append(act_function(Z[0], act_func))
    Z.append(np.dot(w2, A[1]))
    A.append(Z[1])
    return A, Z

# STEP 2: Feed-Backward for AE
def gradW1(a, z, w1, w2, act_f):
    e = a[2]-a[0]
    Cost = np.sum(np.sum(e**2))/(2*e.shape[1])
    del_2 = e
    gW = np.dot(w2.T, del_2) * deriva_act(z[0], act_f)
    gW = np.dot(gW, a[0].T)
    return (gW, Cost)

# Calculate Pseudo-inverse
def pinv_ae(X, A, params):
    p_inverse = int(params[0])
    H = A[0]
    A = (H @ H.T) + ((1 / p_inverse) * np.eye(H.shape[0]))
    A_pinv = np.linalg.pinv(A)
    return (X @ H.T) @ A_pinv

# Update AE's weight via RMSprop
def updW1_rmsprop(w, v, gw, mu):
    beta, eps = 0.9, 1e-8
    v = beta * v + (1 - beta) * gw**2
    gRMS = (1/(np.sqrt(v + eps))) * gw
    w = w - mu*gRMS
    return (w, v)

#Activation function
def act_function(Z:np.ndarray, act_func: int):
    if act_func == 1:
        return np.maximum(0, Z)
    if act_func == 2:
        return np.maximum(0.01 * Z, Z)
    if act_func == 3:
        return np.where(Z > 0, Z, 1 * (np.exp(Z) - 1))
    if act_func == 4:
        lam=1.0507; alpha=1.6732
        return np.where(Z > 0, Z, alpha*(np.exp(Z)-1)) * lam
    if act_func == 5:
        return 1 / (1 + np.exp(-Z))
    
# Derivatives of the activation funciton
def deriva_act(A: np.ndarray, act_func: int):
    if act_func == 1:
        return np.where(A >= 0, 1, 0)
    if act_func == 2:
        return np.where(A >= 0, 1, 0.01)
    if act_func == 3:
        return np.where(A >= 0, 1, 0.01 * np.exp(A))
    if act_func == 4:
        lam=1.0507; alpha=1.6732
        return np.where(A > 0, 1, alpha*np.exp(A)) * lam
    if act_func == 5:
        s = act_function(A, act_func)
        return s * (1 - s)



# Update AE's weight via RMSprop
# def updW1_rmsprop(w,v,gw,mu):
#     beta,eps = 0.9, 1e-8
#     ...    
#     return(w,v)

# Update Softmax's weight via RMSprop
# def updW_sft_rmsprop(w,v,gw,mu):
#     beta, eps = 0.9, 1e-8
#     ...    
#     return(w,v)

# Softmax's gradient
# def gradW_softmax(x,y,a):        
#     ya   = y*np.log(a)
#     ...    
#     return(gW,Cost)
 
# Calculate Softmax
# def softmax(z):
#         exp_z = np.exp(z-np.max(z))
#         return(exp_z/exp_z.sum(axis=0,keepdims=True))

# # save weights SAE and costo of Softmax
# def save_w_dl(W,Ws,cost):    
#     ...


    
