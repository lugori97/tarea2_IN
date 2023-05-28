#Training DL via RMSProp+Pinv

import pandas     as pd
import numpy      as np
import utility    as ut

# # Training miniBatch for softmax
# def train_sft_batch(x,y,W,V,param):
#     costo = []    
#     for i in range(numBatch):   
#         ...
#         ...        
#     return(W,V,costo)

# # Softmax's training via SGD with Momentum
# def train_softmax(x,y,par1,par2):
#     W        = ut.iniW(y.shape[0],x.shape[0])
#     V        = np.zeros(W.shape) 
#     ...
#     for Iter in range(1,par1[0]):        
#         idx   = np.random.permutation(x.shape[1])
#         xe,ye = x[:,idx],y[:,idx]         
#         W,V,c = train_sft_batch(xe,ye,W,V,param)
#         ...
#     return(W,Costo)    



# gets Index for n-th miniBatch
def get_Idx_n_Batch(batch_size, i):
    return np.arange(i*batch_size, i*batch_size + batch_size, dtype=int)

# AE's Training with miniBatch
def train_ae_batch(x,w1,v,w2,param):
    batch_size = param['sae_miniBatch_size']
    numBatch = np.int16(np.floor(x.shape[1]/batch_size))    
    cost= [] 
    mu = param['sae_learning_rate']
    act_f = param['act_function_encoder']
    
    for i in range(numBatch):
        idx = get_Idx_n_Batch(batch_size, i)
        xe = x[:, idx]
        a, z = ut.ae_forward(xe, w1, w2, act_f)
        gw_1, cost = ut.gradW1(a, z, w1, w2, act_f)
        w1, v1 = ut.updW1_rmsprop(w1, v, gw_1, mu)
    return (w1, v1, cost)






# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x, param, w1, w2, v1):
    maxIter = int(param['sae_max_iter'])
    for i in range(1, maxIter):
        xe = x[:, np.random.permutation(x.shape[1])]
        w1, v1, cost = train_ae_batch(xe, w1, v1, w2, param)
        if i % 10 == 0:
            print(f"Iterar-AE: {i},{np.mean(cost)}")
    return (w1)

# SAE's Training 
def train_dl(x,param):
    
    W = []
    
    for i in range(len(param['encoder_nodes'])):
        w1 = ut.iniW(int(param['encoder_nodes'][i]), x.shape[0])
        w2 = ut.iniW(x.shape[0], int(param['encoder_nodes'][i]))
        v1 = np.zeros_like(w1)
        w1 = train_ae(x, param, w1, w2, v1)
        z = np.dot(w1, x)
        x = ut.act_function(z, param['act_function_encoder'])
        W.append(w1)
    return (W, x)

    #for hn in range(5, len(param)):
   # for i in range(len(p_sae['encoder_nodes'])):        
   #    w1       = train_ae(x,i,p_sae)  
   #   Z = w1 * x 
   #  x        = ut.act_function(Z,p_sae['act_function_encoder'])
   # W.append(w1)
    return(W,x) 

#load Data for Training
def load_data_trn():
    xe= np.loadtxt('X_train.csv', delimiter=',', dtype=float)
    ye = np.loadtxt('y_train.csv', delimiter=',', dtype=float)
    return(xe.T,ye.T)

# Beginning ...
def main():
    p_sae,p_sft = ut.load_config()         
    xe,ye       = load_data_trn() 
    W,Xr        = train_dl(xe,p_sae)         
    # Ws, cost    = train_softmax(Xr,ye,p_sft,p_sae)
    # ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

