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

import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import GPy

def find_modal(samples, linspace_num: int=1000):

    xmin = samples.min()*0.9
    xmax = samples.max()*1.1
    xs = np.linspace(xmin,xmax,linspace_num)
    
    try:

        kde = gaussian_kde(samples)
        density = kde.pdf(xs)
        argmax = np.argmax(density)
        modal = xs[argmax]

        return modal
    
    except:
        
        return None

def gp_model(X: np.array, 
             y: np.array, 
             is_mcmc: bool=False, 
             num_samples: int=1000,
             hmc_iters: int=2,
             linspace_num: int=1000) -> GPy.models.GPRegression:
    
    dim = X.shape[-1]

    if is_mcmc:        

        factor = 10.0 # TODO: clever factor for numerical issues in HMC train

        kern = GPy.kern.RBF(input_dim=dim, ARD=True)
        model = GPy.models.GPRegression(factor*X,y,kernel=kern.copy())    

        # TODO: automate prior for RBF variance
        model.kern.variance.set_prior(GPy.priors.Gamma.from_EV(0.1,0.1),warning=False)

        lengthscales = {}
        for i in range(dim):
            kde = gaussian_kde(X[:,i])
            mean = quad(lambda x: x * kde.pdf(x), a=-np.inf, b=np.inf)[0]
            var = quad(lambda x: x**2 * kde.pdf(x), a=-np.inf, b=np.inf)[0] - mean**2
            lengthscales[i] = np.sqrt(var)
            model.kern.lengthscale[[i]].set_prior(GPy.priors.Gamma.from_EV(lengthscales[i],lengthscales[i]/2),warning=False) # data variance as length scale

        hmc = GPy.inference.mcmc.HMC(model)
        samples = hmc.sample(num_samples=num_samples,hmc_iters=hmc_iters)

        modals = {}          
        for i in range(samples.shape[-1]):
            modal = find_modal(samples[:,i],linspace_num)
            if modal is not None:
                modals[i] = modal

        kern = GPy.kern.RBF(input_dim=dim, ARD=True)
        model = GPy.models.GPRegression(X,y,kernel=kern.copy())    

        if (0 in modals) and (dim-1 in modals):            
            model.rbf.variance = modals[0]/factor**2
            model.Gaussian_noise.variance = modals[dim-1]/factor**2
        else:
            model = None
            
        for i in range(dim):
            if i in modals:
                model.rbf.lengthscale[i] = modals[1+i]/factor
            else:
                model.rbf.lengthscale[i] = lengthscales[i]

    else:

        kern = GPy.kern.RBF(input_dim=dim, ARD=True)
        model = GPy.models.GPRegression(X,y,kernel=kern.copy())    
        
        # model.optimize_restarts(num_restarts=10,optimizer='lbfgs',verbose=False)
        model.optimize(optimizer='lbfgs',messages=False)
        
    return model
