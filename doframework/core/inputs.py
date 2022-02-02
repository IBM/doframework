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

import scipy.stats
import numpy as np
import random
import string
import logging
from typing import Optional

from doframework.core.utils import logger

def parse_dim(sim_input: dict) -> int:
    
    dim = -1
    
    if ('f' in sim_input) and ('vertices' in sim_input['f']):
        
        vertices = sim_input['f']['vertices']

        if ('position' in vertices) or ('range' in vertices):
            if ('position' in vertices) and isinstance(vertices['position'], list):
                if all([isinstance(p, list) for p in vertices['position']]):
                    dims = [len(p) for p in vertices['position']]
                    if min(dims)==max(dims):
                        dim = min(dims)
            elif ('range' in vertices) and isinstance(vertices['range'], list):
                if all([isinstance(r, list) for r in vertices['range']]):
                    if all([len(r)==2 for r in vertices['range']]):
                        dim = len(vertices['range'])
            else:
                pass            
        else:
            pass
        
    elif ('f' in sim_input) and ('polyhedrons' in sim_input['f']):
        
        polys = sim_input['f']['polyhedrons']
        dim = np.vstack(polys).shape[-1]
            
    else:
        pass        
        
    return dim

def parse_vertex_num(sim_input: dict) -> int:
    
    num = -1
    
    if ('f' in sim_input) and ('vertices' in sim_input['f']): 
        
        vertices = sim_input['f']['vertices']
        
        if ('position' in vertices):            
            if isinstance(vertices['position'],list):        
                num = len(vertices['position'])            
        elif ('num' in vertices):        
            if isinstance(vertices['num'],int):        
                num = vertices['num']            
    else:        
        pass
    
    return num

#### TODO: assert if f[vertices] or omega has vertices and omega[vertices] has position then num of points > dim
####       assert if f[vertices] has position, then omega has vertices and omega[vertices] has either num or position
####       assert if f[values] has coeffs, then omega has vertices and omega[vertices] has either num or position
@logger
def legit_input(sim_input: dict, 
                logger_name: Optional[str]=None, 
                is_raised: Optional[bool]=False):
    '''
    Test input validity. 
    
            Parameters:
                    sim_input (dict): input. 
    '''
    
    
    dim = parse_dim(sim_input)
    vertex_num = parse_vertex_num(sim_input)
    
    assert 'f' in sim_input, \
    'Input missing function info under \'f\' field.'

    assert 'omega' in sim_input, \
    'Input missing feasibility region info under \'omega\' field.'

    assert 'data' in sim_input, \
    'Input missing data info under \'data\' field.'

    f = sim_input['f']

    assert dim>0, \
    'Problem dimension could not be deduced from input f vertices.\
    Vertices attributes range and/or position are missing or incorrect.'

    assert vertex_num>0, \
    'Vertex number could not be deduced from input f vertices.\
    Vertex number either missing or wrong type.'

    assert any([val in f['values'] for val in ['coeffs','evals','range']]), \
    'f values missing either coeffs, evals, or range field under \'f\'.'
    
    if 'coeffs' in f['values']:

        assert isinstance(f['values']['coeffs'],list), \
        'Expecting a list of numbers under f.values.coeffs.'

        assert len(f['values']['coeffs'])==dim+1, \
        f'Number of f.values.coeffs must match dimension deduced from f.vertices. Dimension deduced is {dim}.'

    elif 'evals' in f['values']:
        
        assert isinstance(f['values']['evals'],list), \
        'Expecting a list of numbers under f.values.evals.'

        assert len(f['values']['evals'])==vertex_num, \
        f'Number of f.values.evals must match f.vertices.num {vertex_num}.'

    else:
        
        assert 'range' in f['values'], \
        'Expecting a range of values for f in the absence of coeffs or evals.'
        
        assert isinstance(f['values']['range'],list), \
        'Expecting a list of numbers under f.values.range.'

        assert len(f['values']['range'])==2, \
        'f.values.range requires [min,max].'

        assert f['values']['range'][0]<f['values']['range'][1], \
        'f.values.range requires [min,max] where min<max.'

    omega = sim_input['omega']

    assert all([val in omega for val in ['ratio','hypothesis','scale']]), \
    'Input missing ratio, hypothesis, and scale as fields under \'omega\'.'

    assert isinstance(omega['ratio'],float), \
    'omega.ratio must be a float.'

    assert omega['hypothesis'] in [name for name, _ in scipy.stats._distr_params.distcont], \
    'omega.hypothesis must be a continuous distribution from scipy.stats.'

    assert isinstance(omega['scale'],float), \
    'omega.scale must be a float.'

    data = sim_input['data']

    assert all([val in data for val in ['noise','hypothesis','policy_num','scale']]), \
    'Input missing noise, hypothesis, policy_num, and scale as fields under \'data\'.'

    assert data['hypothesis'] in scipy.stats._multivariate.__all__, \
    'data.hypothesis must be a multivariate distribution from scipy.stats.'

# NOTE: database tables assume the default N=8
def generate_id(N: int=8) -> str:
    
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))


def setup_logger(logger_name: str, log_file: str, level=logging.DEBUG, to_stream: bool=False):
    
    l = logging.getLogger(logger_name)    
    l.setLevel(level)
    
    #### stop matplotlib from dumping logs
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.addHandler(fileHandler)
    
    if to_stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        l.addHandler(streamHandler)    