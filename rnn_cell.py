import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax import random

import time
import sys

from helpers import generate_ou_process, plot_ou_process

# Generate key which is used to generate random numbers
key = random.PRNGKey(1)

from jax.nn import sigmoid, softmax, log_softmax, one_hot
from jax.nn.initializers import glorot_normal, normal

from functools import partial
from jax import lax
from jax.experimental import stax
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax)

def GRU(out_dim, W_init=glorot_normal(), b_init=normal()):
    def init_fun(rng, input_shape):
        """ Initialize the GRU layer for stax """
        hidden = np.zeros((1, out_dim))

        k1, k2, k3 = random.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)
        
        #softmax parameters
        k1, k2, k3 = random.split(rng, num=3) 
        sm_W, sm_b = (W_init(k1, (out_dim,input_shape[2])),b_init(k3, (input_shape[2],)),)
        
        rng, subkey = random.split(rng)  
        
        # Input dim 0 represents the batch dimension
        # Input dim 1 represents the time dimension (before scan moveaxis)
        output_shape = (input_shape[0], input_shape[1], out_dim)
        return (output_shape,
            (hidden,
             (update_W, update_U, update_b),
             (reset_W, reset_U, reset_b),
             (out_W, out_U, out_b),(sm_W, sm_b)),rng)
    
    def sample(params,Nsi,Nq,key): 
        """ Sample all the conditionals parameterized by the rnn """
        h = lax.stop_gradient(np.tile(params[0], [Nsi.shape[0],1])) # initialize the hidden state at t = 0
       
        def sample_scan(params, tup,x):
            """ Perform single step update of the network """
            _, (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
                out_W, out_U, out_b),(sm_W, sm_b) = params
            hidden = tup[3]
            logP = tup[2]
            key = tup[0] 
            inp = tup[1]  

            update_gate = sigmoid(np.dot(inp, update_W) +
                                  np.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(np.dot(inp, reset_W) +
                                 np.dot(hidden, reset_U) + reset_b)
            output_gate = np.tanh(np.dot(inp, out_W)
                                  + np.dot(np.multiply(reset_gate, hidden), out_U)
                                  + out_b)
            output = np.multiply(update_gate, hidden) + np.multiply(1-update_gate, output_gate)
            hidden = output
            logits = np.dot(hidden,sm_W) + sm_b

            key, subkey = random.split(key)
            
            samples = random.categorical(subkey, logits, axis=1, shape=None) # sampling the conditional
            samples = one_hot(samples,sm_b.shape[0]) 
            log_P_new = np.sum(samples*log_softmax(logits),axis=1)
            log_P_new = log_P_new + logP 
             
            
            #print("shape of logP",logP.shape)
            return (key,samples,log_P_new,output),samples
        
        Ns = Nsi.shape[0] 
        logP = np.zeros((Ns,)) # initialize logP = 0

        sample_im1 = np.zeros((Ns,params[4][1].shape[0])) # first input cell is zero. params[4][1].shape[0] contains the hilbert space dim

        f = partial(sample_scan, params)
        (key_new,samples_new,logP_new,hidden_new),out_new = lax.scan(f, (key,sample_im1,logP,h),None,Nq)
               
        out_new = np.transpose(out_new,[1,0,2])   # reshape to [bsize,Nqubits,loc hilbert space dim]

        return out_new,logP_new,key_new #()    

    def apply_fun(params, inputs, **kwargs):
        """ Loop over the time steps of the input sequence """
        h = lax.stop_gradient(np.tile(params[0], [inputs.shape[0],1]))
        logP = np.zeros((inputs.shape[0],inputs.shape[2],))  
        def apply_fun_scan(params, tup, inp):
            """ Perform single step update of the network """
            _, (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
                out_W, out_U, out_b),(sm_W, sm_b) = params
            hidden = tup[0]
            logP = tup[1]

            update_gate = sigmoid(np.dot(inp, update_W) +
                                  np.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(np.dot(inp, reset_W) +
                                 np.dot(hidden, reset_U) + reset_b)
            output_gate = np.tanh(np.dot(inp, out_W)
                                  + np.dot(np.multiply(reset_gate, hidden), out_U)
                                  + out_b)
            output = np.multiply(update_gate, hidden) + np.multiply(1-update_gate, output_gate)
            hidden = output
            logP = log_softmax(np.dot(hidden,sm_W) + sm_b)

            return (hidden,logP),(hidden,logP)
        
        Nq = inputs.shape[1]
        zeroin = np.zeros((inputs.shape[0],1,inputs.shape[2])) # The first input to the RNN is always zero
        input_shift = np.concatenate((zeroin,inputs),axis=1)   # Concatenate the zero input for the first call to the input
        input_shift = input_shift[:,0:Nq,:] # cut the last spin of the  shifted input  

        # Move the time dimension to position 0
        input_shift = np.moveaxis(input_shift, 1, 0) # scan needs "time" to be the first dimension to loop over
        f = partial(apply_fun_scan, params)
        _,out_new = lax.scan(f, (h,logP), input_shift)

        logP_inputs = np.sum(np.transpose(out_new[1],(1,0,2))*inputs,axis=[1,2]) # transpose time <--> batch, then sum over the non zero elements of logP 
        
        return out_new,logP_inputs #( hidden state at each time step[time, batch, hiddenunites], logP at each time step[time, batch, local_hilbert_dim], logP[])

    return init_fun, apply_fun,sample

data_train = onp.loadtxt("train.txt")

num_dims = 16             # Number of  timesteps/ number of qubits
batch_size = 400           # Batchsize 
num_hidden_units = 100     # GRU cells in the RNN layer
local_hilbert_dim = 2      # dimension of the local Hilbert space
Nqubits = num_dims

init_fun, gru_rnn, sample = GRU(num_hidden_units)

_, params,key = init_fun(key, (batch_size, num_dims, local_hilbert_dim))

@jit
def sample_o(shapes,key):
    Nq = shapes.shape[0]
    Ns = batch_size*shapes.shape[1] 
    Nsi = np.zeros(Ns) 
    configs,logP,key = sample(params,Nsi,Nq,key)
    return configs,logP,key 

def mle_loss(params,inputs):
    _,lP = gru_rnn(params,inputs)
    return -np.mean(lP)
    
@jit
def update(params, x, opt_state):
    """ Perform a forward pass, calculate the MSE & perform a SGD step. """
    loss, grads = value_and_grad(mle_loss)(params, x)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, loss

from jax.scipy.special import logsumexp
from jax.experimental import optimizers

step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

n_samples = data_train.shape[0]
num_batches = int(n_samples/batch_size)

train_loss_log = []
start_time = time.time()

counter = 0 
bcount = 0 

for epoch in range(2):
    ept = onp.random.permutation(data_train)  
    for i in range(num_batches):
        if bcount*batch_size + batch_size>=n_samples:
            bcount = 0
            ept = onp.random.permutation(data_train)
            
        x_in = ept[ bcount*batch_size: bcount*batch_size+batch_size,:]
        x_in = np.reshape(x_in,[batch_size,Nqubits,local_hilbert_dim])
        bcount=bcount+1  
        params, opt_state, loss = update(params, x_in, opt_state)
    print("epoch ",epoch,"th done")
    avlp=0.0  
for i in range(num_batches):
    print(i) 
    x = ept[ i*batch_size: i*batch_size+batch_size,:]
    avlp = avlp + mle_loss(params,np.reshape(x,[batch_size,Nqubits,local_hilbert_dim]))
avlp=avlp/num_batches
print("ave logP train",avlp)   

sys.exit(0)

