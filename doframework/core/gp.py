#
# Copyright IBM Corporation 2022
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import GPy

def plot_posteriors(samples: np.array, labels: List[str], **kwargs):    

    '''
    Plot hyper-parameter posteriors following HMC. By convention, first column of samples is RBF kernel variance chain. 
    The last column of samples is the RBF kernel noise. The middle columns of samples are ARD length scale parameters.
    
            Parameters:
                    samples (np.array): hyper-parameter HMC chains.
                    labels (List[str]): sample column labels for graph legends.
    '''        
    
    import matplotlib.pyplot as plt

    num_of_plots = 3
    num_of_posteriors = samples.shape[-1]
    dim = num_of_posteriors - num_of_plots + 1

    assert len(labels) == num_of_posteriors, \
    f'Expected {num_of_posteriors} plot labels, but received {len(labels)}.'

    fig_width = kwargs['fig_width'] if 'fig_width' in kwargs else 10
    fig_length = kwargs['fig_length'] if 'fig_length' in kwargs else 3

    _, axs = plt.subplots(num_of_plots,1,figsize=(fig_width,num_of_plots*fig_length))
    cmap = plt.cm.get_cmap(name='Accent',lut=num_of_posteriors)

    for i, J in zip(range(num_of_plots),[[0],[1+j for j in range(dim)],[dim+1]]):

        modals = []

        for j in J:
            
            s = samples[:,j]
            xmin = s.min()*0.9
            xmax = s.max()*1.1
            xs = np.linspace(xmin,xmax,1000)

            if ('kdes' in kwargs) and (j in kwargs['kdes']):
                kde = kwargs['kdes'][j]
            else:
                kde = gaussian_kde(s)

            if ('modals' in kwargs) and (j in kwargs['modals']):
                modal = kwargs['modals'][j]
            else:
                density = kde.pdf(xs)
                argmax = np.argmax(density)
                modal = xs[argmax]

            modals.append(modal)            
            modal_density = kde.pdf(modal)[0]

            axs[i].plot([modal,modal],[0,modal_density],ls='--',color=cmap(j)) 
            axs[i].plot(xs,kde(xs),label=labels[j],lw=3,color=cmap(j))

        ticks = np.sort(np.hstack([np.around(np.array(modals),2), axs[i].get_xticks()]))
        axs[i].set_xticks(ticks[1:-1]) # ticks[1:-1]
        axs[i].tick_params(axis="x", rotation=90, labelsize=12)
        axs[i].legend()

    plt.tight_layout()

def find_modal(samples, linspace_num: int=1000):

    xmin = samples.min()*0.9
    xmax = samples.max()*1.1
    xs = np.linspace(xmin,xmax,linspace_num)
    
    try:

        kde = gaussian_kde(samples)
        density = kde.pdf(xs)
        argmax = np.argmax(density)
        modal = xs[argmax]

        return modal, kde
    
    except:
        
        return None, None

def gp_model(X: np.array, 
             y: np.array, 
             is_mcmc: bool=False, 
             num_samples: int=1000,
             hmc_iters: int=2,
             plot_kernel_posteriors: bool=False,
             linspace_num: int=1000) -> GPy.models.GPRegression:
    
    dim = X.shape[-1]

    if is_mcmc:        

        factor = 10.0 # factor for numerical issues in HMC train

        kern = GPy.kern.RBF(input_dim=dim, ARD=True)
        model = GPy.models.GPRegression(factor*X,y,kernel=kern.copy())    

        # automate prior for RBF variance
        model.kern.variance.set_prior(GPy.priors.Gamma.from_EV(0.1,0.1),warning=False)

        lengthscales = {}
        kdes = {}
        for i in range(dim):
            kde = gaussian_kde(X[:,i])
            kdes[i+1] = kde
            mean = quad(lambda x: x * kde.pdf(x), a=-np.inf, b=np.inf)[0]
            var = quad(lambda x: x**2 * kde.pdf(x), a=-np.inf, b=np.inf)[0] - mean**2
            lengthscales[i] = np.sqrt(var)
            model.kern.lengthscale[[i]].set_prior(GPy.priors.Gamma.from_EV(lengthscales[i],lengthscales[i]/2),warning=False)

        hmc = GPy.inference.mcmc.HMC(model)
        samples = hmc.sample(num_samples=num_samples,hmc_iters=hmc_iters)

        modals = {}
        kdes = {}
        for i in range(samples.shape[-1]):
            modal, kde = find_modal(samples[:,i],linspace_num)
            if (modal is not None) and (kde is not None):
                modals[i] = modal
                kdes[i] = kde

        kern = GPy.kern.RBF(input_dim=dim, ARD=True)
        model = GPy.models.GPRegression(X,y,kernel=kern.copy())    

        if (0 in modals) and (dim-1 in modals):            
            model.rbf.variance = modals[0]/factor**2
            model.Gaussian_noise.variance = modals[dim-1]/factor**2
        else:
            raise ValueError('HMC failed. Possible unsuitable priors on kernels parameters leading to repetative samples.')

        for i in range(dim):
            if i in modals:
                model.rbf.lengthscale[i] = modals[1+i]/factor
            else:
                model.rbf.lengthscale[i] = lengthscales[i]

        if plot_kernel_posteriors:
            labels = ['RBF kernel variance']+ [f'RBF kernel lengthscale[x{i}]' for i in range(dim)] + ['RBF kernel noise']
            plot_posteriors(samples, labels, modals=modals, kdes=kdes)

    else:

        kern = GPy.kern.RBF(input_dim=dim, ARD=True)
        model = GPy.models.GPRegression(X,y,kernel=kern.copy())    

        # model.optimize_restarts(num_restarts=10,optimizer='lbfgs',verbose=False)
        model.optimize(optimizer='lbfgs',messages=False)

    return model