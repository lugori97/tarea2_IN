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

# AE's Training with miniBatch
def train_ae_batch(x,w1,v,w2,param):
    numBatch = np.int16(np.floor(x.shape[1]/param[0]))    
    cost= [] 
    for i in range(numBatch):                
        ...             
    return(w1,v,cost)

# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x,i,p_sae):

    W = ut.iniW()

    for i in range(len(p_sae['sae_max_iter'])):
        w2 = 
        
    return W

# SAE's Training 
def train_dl(x,p_sae):
    
    W = []
    
    for i in range(len(p_sae['encoder_nodes'])):        
        w1       = train_ae(x,i,p_sae)  
        Z = w1 * x 
        x        = ut.act_function(Z,p_sae['act_function_encoder'])
        W.append(w1)
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
    print(W)      
    # Ws, cost    = train_softmax(Xr,ye,p_sft,p_sae)
    # ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

