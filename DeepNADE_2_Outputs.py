#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Jan 28 2021

@author: Alex Lidiak, Caleb Sanders
This model takes as input a FFNN (N inputs, 2 outputs) and converts it into a 
Deep NADE model. Options for order-agnostic and order averaging that should 
increase the model's accuracy. 

For more information on DeepNADEs and the inspiration for this lib see:
https://arxiv.org/abs/1605.02226, Uria, B., Côté, M. A., Gregor, K., Murray, I.
, & Larochelle, H. (2016). Neural autoregressive distribution estimation. 
The Journal of Machine Learning Research, 17(1), 7184-7220.

"""

def update_grads(network, e_grad):
    """
    Update network parameter gradients to the calculated energy gradient
    Run before a call to optimizer.step()

    Args: 
            e_grad: iterable, 1D energy gradient with a length of the number of network parameters
    """

    e_grad_index = 0
    params = list(network.parameters())

    with torch.no_grad():
        
        for param in params:
            
            # gets the energy gradient values for the given parameter
            e_grad_slice = e_grad[e_grad_index : e_grad_index+param.nelement()]
            e_grad_index += param.nelement()

            # reshape the slice of the gradient to match the shape of the parameter
            e_grad_tensor = torch.reshape(e_grad_slice, param.size())
            
            param.grad.copy_(e_grad_tensor) 

def calculate_epsilon(model, s, psi_s, B, J):
        """
        Calculates the energy contribution (epsilon) of a given state for a desired hamiltonian 

        Args: 
            model: DeepNADE model 
            s: tensor, sampled state 
            B: int, sigma_x term activation
            J: int, sigma_z term activation 

        Returns: 
            epsilon: double, epsilon contribution for the given state 
        """

        N = len(s)

        # epsilon(s) = sum(s_i * s_i+1) + B/psi_s * sum(psi_s_prime)
        
        z_term = 0
        psi_s_prime_sum = 0 #sum of the wavefunction coefficients resulting from sigma_x acting on each qubit
        for i in range(N):
            
            if i == N-1:
                z_term += s[0]*s[i]
            else:
                z_term += s[i]*s[i+1]

            # calculate the sum of psi_s_prime for the sigma_x term
            s_prime = s.clone().detach()
            s_prime[i] = -1*s_prime[i]
            psi_s_prime,_,_ = model(x=s_prime.unsqueeze(0)) 
            psi_s_prime_sum += float(psi_s_prime)
        
        x_term = psi_s_prime_sum/float(psi_s)

        epsilon = float(J*z_term + B*x_term)
        
        return epsilon

def calculate_Ok(sample_grads):
    """
    Calculate the O_k's for a system. 
    O_k matrix is a 2D matrix with shape (n_spins x n_params x 1).

    Args: sample_grads, list of sample gradient tensors for each state
    Returns: O_k, tensor of (1 x n_params) lists for each state 
    """

    O_k = []

    for spin_grad in sample_grads:

        # mini_arr contains the O_k values for a given spin state  
        mini_tens = torch.tensor([])

        #calculate the O_k value for each parameter 
        for grad in spin_grad:

            O_k_s = grad.flatten()
            mini_tens = torch.cat((mini_tens, grad.flatten()))

        O_k.append(mini_tens)

    O_k = torch.stack(O_k)
    return O_k


class DeepNADE(nn.Module): # takes a FFNN model as input
            
    def __init__(self, model): # TODO: add ordering here as class property?
        super(DeepNADE, self).__init__()
        
        self.model = model
        self.D = self.model[0].in_features # input layer size
        self.M = self.model[-2].out_features # output layer size
        self.evals = [0,1]
            
    def forward(self, N_samples=None, x=None):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if N_samples is None and x is None: 
            raise ValueError('Must enter samples or the number of samples to' \
                             ' be generated')
            
        if N_samples is None and x is not None: 
            N_samples, sample = x.shape[0], False

        if N_samples is not None and x is None: 
            sample = True 
            x = torch.zeros([N_samples,self.D],dtype=torch.float).to(device)
            
        # the full prob is a product of the conditionals, making a running 
        # product easy
        WAV = torch.ones([N_samples]).to(device) # if multiplying
        order = np.arange(0,self.D) # sequential autoregressive ordering 
        
        params = list(self.parameters())
        grads = [ [] for i in range(0,len(x)) ] #allocate gradient storeage
        for sample in range(0,len(x)):
          for param in params:
            grads[sample].append(np.zeros_like(param.detach().numpy()))
        
        for d in range(self.D):

            #print("GENERATING BIT {}".format(d+1))
                
            # masks enforce the autoregressive property
            mask=torch.zeros_like(x)
            mask[:,order[0:(d)]]=1 

            if not hasattr(self.model,'autograd_hacks_hooks'):             
              autograd_hacks.add_hooks(self.model)
            
            # L2 normalization of masked output
            out = F.normalize(self.model(mask*x), 2)
            
            # select the wav coeffs corresponding to neg an pos bits 
            psi_pos = out[:,0].squeeze()
            psi_neg = out[:,1].squeeze()
            
            # Sampling probability is determined by the separate conditionals
            if sample:
                
                # sampling routine according to wav**2:
                m = torch.distributions.Bernoulli(psi_pos**2).sample()

                # convert bit values of 0 to -1 
                for bit in m:
                  if bit == 0.: bit.copy_(torch.tensor(-1, dtype=torch.float))
                
                # update sample tensor
                x[:,d] = m

                # Accumulate PPSI based on which state (s) was sampled
                selected_wavs = torch.where(x[:,d] > 0, psi_pos, psi_neg) 
                WAV = WAV*selected_wavs

                # PER-SAMPLE GRADIENT CALCULATION 
                # eval_grads stores backpropagation values for out1 and out2.
                # eval_grads[0] are the grads for backpropping out1 for all 
                # samples, eval_grads[1] are the grads for out2. 
                eval_grads = [ [[]]*len(params) for outputs in range(len(model.evals))]

                # Store the per-output grads in eval_grads
                for output in range(len(self.evals)):

                  # backpropagate the current output (out1 or out2)
                  out[:,output].mean(0).backward(retain_graph=True)

                  # compute gradients for all samples 
                  autograd_hacks.compute_grad1(self.model)
                  autograd_hacks.clear_backprops(self.model)
                  
                  # store the calculated gradients for all samples 
                  for param in range(len(params)):
                    eval_grads[output][param] = params[param].grad1.numpy()
                
                # select gradients that correspond to sampled bits
                for sample in range(len(x)):
                  for param in range(len(params)):
                    if m[sample] > 0:
                      grads[sample][param] += eval_grads[0][param][sample]/selected_wavs[sample].detach().numpy()
                    else:
                      grads[sample][param] += eval_grads[1][param][sample]/selected_wavs[sample].detach().numpy()

            else:

              # Accumulate PPSI based on which state (s) was sampled
              selected_wavs = torch.where(x[:,d] > 0, psi_pos, psi_neg) 
              WAV = WAV*selected_wavs
                
        return WAV.detach(), x.detach(), grads
