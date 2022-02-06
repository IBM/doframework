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

import os
import getpass
import logging
import argparse
from typing import Optional, Tuple
import yaml
import json
from datetime import datetime
import itertools as it
import numpy as np
from scipy.stats import random_correlation, uniform, beta
from scipy.spatial import ConvexHull

from doframework.core.inputs import setup_logger, legit_input, parse_dim, parse_vertex_num, generate_id
from doframework.core.triangulation import Process, triangulation, pts_iterator
from doframework.core.utils import flatten, sample_standard_simplex
from doframework.core.pwl import PWL, Polyhedron, PolyLinear, argopt
from doframework.core.sampler import sample_f_values

def calculate_objectives(meta_input: dict, args: dict) -> int:
    f = meta_input['f']
    omega = meta_input['omega']
    objectives = 1 if ('position' in f['vertices']) and \
                    ('vertices' in omega) and \
                    ('position' in omega['vertices']) and \
                    (('evals' in f['values']) or ('coeffs' in f['values'])) else args.objectives
    return objectives

#### TODO: remove the dependence on poly to improve performance
def get_omega_P(vertex_input: dict, poly, logger_name: Optional[str]=None,is_raised: Optional[bool]=False) -> np.array:

    if 'position' in vertex_input:
        points = np.atleast_2d(np.array(vertex_input['position']))
        points = points[poly.isin(points)]
        if logger_name:
            log = logging.getLogger(logger_name)
            log.info('Extracted Omega vertices from input. Points out of Supp(f) were removed.\n{}.'.format([list(row) for row in points]))
    else:
        vertex_num = vertex_input['num']
        if 'range' in vertex_input:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('Sampling {} vertices for Omega within given range.'.format(vertex_num))
            vertex_range_points = np.vstack(list(map(np.array, it.product(*vertex_input['range']))))
            points = np.vstack(
                list(it.islice(
                    filter(lambda point: Polyhedron(vertex_range_points).isin(point)[0],
                           pts_iterator(poly,1))
                    ,vertex_num
                ))                
            )
        else:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('Sampling {} vertices for Omega within Supp(f).'.format(vertex_num))
            points = poly.sample(vertex_num)
    try:
        hull = ConvexHull(points,qhull_options='QJ')
        return hull.points[hull.vertices,:]
    except Exception as e:
        if logger_name:
            log = logging.getLogger(logger_name)
            log.error('QHull failure on points: {}.'.format([list(row) for row in points]))
        if is_raised: raise e
        
def generate_objective(meta_input: dict, meta_name: str, logger_name: Optional[str]=None, is_raised: Optional[bool]=False, **kwargs) -> Tuple[dict, str]:
    '''
    Generate PWL objective targets from meta input.
    
            Parameters:
                    meta_input (dict): Meta data for objective target generation. 
                    meta_name (str): Name of meta data file.
                    logger_name (str): Name of logger object. Will write logs defined by this logger name. Default None [no logging].
                    is_raised (bool): Raise errors. Default False.
                    
            Returns:
                    Returns the objective target output dictionary and the name of the objective file.
    '''
    
    output_prefix = 'objective'
    output_suffix = 'json'

    objective_id = generate_id()

    f = meta_input['f']
    omega = meta_input['omega']
    data = meta_input['data']    
    dim = parse_dim(meta_input)

    output = {}
    output['f'] = {}
    output['omega'] = {}
    output['data'] = {}
    output['optimum'] = {}

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info('Started output for objective ID {}.'.format(objective_id))

    if 'position' in f['vertices']:
        
        f_points = np.atleast_2d(np.array(f['vertices']['position']))
        f_supp_range = np.array([[f_points[:,i].min(),f_points[:,i].max()] for i in range(dim)])
        f_hull = ConvexHull(f_points,qhull_options='QJ')
        f_P = f_hull.points[f_hull.vertices,:]
        if logger_name:
            log = logging.getLogger(logger_name)
            log.info('Extracted Supp(f) vertices from input.')
        f_poly = Polyhedron(f_P) #### TODO: remove the dependence on f_poly to improve performance
        f_Ps = [f_P]
            
        omega_vertices = omega['vertices'] # in this case, omega must have 'vertices', which must have either 'position' or 'num'
        omega_P = get_omega_P(omega_vertices,f_poly,logger_name,is_raised)
        omega_Ps = [omega_P]
        
        if 'coeffs' in f['values']:
            f_coeffs = f['values']['coeffs']
            f_V = np.pad(f_P,[(0,0),(0,1)],constant_values=1) @ f_coeffs
            omega_V = np.pad(omega_P,[(0,0),(0,1)],constant_values=1) @ f_coeffs        
        elif 'evals' in f['values']:
            f_V = np.array(f['values']['evals'])[f_hull.vertices]
            omega_V = PolyLinear(f_P,f_V).evaluate(omega_P)        
        else:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('Sampling {} values for Supp(f) vertices.'.format(f_P.shape[0]))
            f_V = sample_f_values(f['values']['range'],f_P.shape[0])
            omega_V = PolyLinear(f_P,f_V).evaluate(omega_P) 
        
        f_Vs = [f_V]
        omega_Vs = [omega_V]
                
    else:

        f_supp_range = np.atleast_2d(np.array(f['vertices']['range']))
        f_P = np.vstack(list(map(np.array, it.product(*f['vertices']['range']))))        
        
        if 'coeffs' in f['values']:

            f_poly = Polyhedron(f_P) #### TODO: remove the dependence on f_poly to improve performance
            f_Ps = [f_P]    
            omega_vertices = omega['vertices'] # omega must have vertices in this case
            omega_P = get_omega_P(omega_vertices,f_poly,logger_name=logger_name,is_raised=is_raised)        
            omega_Ps = [omega_P]
            f_coeffs = f['values']['coeffs']
            f_V = np.pad(f_P,[(0,0),(0,1)],constant_values=1) @ f_coeffs
            omega_V = np.pad(omega_P,[(0,0),(0,1)],constant_values=1) @ f_coeffs                
            f_Vs = [f_V]
            omega_Vs = [omega_V]

        else:

            ratio = omega['ratio']
            f_range = f['values']['range']
            vertex_num = parse_vertex_num(meta_input)    
            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('Running triangulation algorithm: N={}, ratio={}.'.format(vertex_num,ratio))

            p = kwargs['p'] if 'p' in kwargs else 0.5
            q = kwargs['q'] if 'q' in kwargs else 1
            
            # default prob: use upper bound on number of simplices vertex_num-dim
            regularization_min_prob_default = beta(1,vertex_num-dim-1).cdf(1/(vertex_num-dim) - beta(1,vertex_num-dim-1).std())
            regularization_min_prob = kwargs['regularization_min_prob'] if 'regularization_min_prob' in kwargs else regularization_min_prob_default

            p_process = Process(uniform(scale=p))
            q_process = Process(uniform(scale=q))

            f_supp, omega_supp = triangulation(f_supp_range, f_range, ratio, vertex_num, dim, p_process, q_process, regularization_min_prob, logger_name=logger_name, is_raised=is_raised)            
            f_Ps = flatten([[leaf.poly.points for leaf in tree.leaves()] for tree in flatten(f_supp)])
            f_Vs = flatten([[leaf.poly.values for leaf in tree.leaves()] for tree in flatten(f_supp)])
            omega_Ps = flatten([[leaf.poly.points for leaf in tree.leaves()] for tree in flatten(omega_supp)])
            omega_Vs = flatten([[leaf.poly.values for leaf in tree.leaves()] for tree in flatten(omega_supp)])  

    output['f']['polyhedrons'] = [[list(point) for point in P] for P in f_Ps]
    output['f']['values'] = [list(V) for V in f_Vs]

    pwl = PWL(f_Ps,f_Vs)
    f_supp_scale = np.power(pwl.volume(),1/dim) # width parameter, approx of f domain diameter 

    omega_hull = ConvexHull(np.vstack(omega_Ps))
    omega_locs = omega_hull.points[omega_hull.vertices,:] # this may shift original vertices on the order of 1e-8
    omega_scales = np.random.rand(*omega_locs.shape)*omega['scale']*f_supp_scale

    output['omega']['polyhedrons'] = [[list(point) for point in P] for P in omega_Ps]
    output['omega']['hypothesis'] = omega['hypothesis'] if 'hypothesis' in omega else 'norm'
    output['omega']['locs'] = [list(row) for row in omega_locs]
    output['omega']['scales'] = [list(row) for row in omega_scales]
    
    policies = pwl.sample(data['policy_num'])
    eigenvals = np.vstack([sample_standard_simplex(dim)*dim for _ in range(data['policy_num'])])
    corrs = [random_correlation.rvs(e) for e in eigenvals]
    sigmas = [np.random.rand(dim)*data['scale']*f_supp_scale for _ in range(data['policy_num'])]
    covs = [np.diag(sigma) @ corr @ np.diag(sigma) for corr, sigma in zip(corrs,sigmas)]
    weights = sample_standard_simplex(data['policy_num'])

    output['data']['N'] = data['N']
    output['data']['hypothesis'] = data['hypothesis'] if 'hypothesis' in data else 'multivariate_normal'
    output['data']['policies'] = [list(row) for row in policies]
    output['data']['covariances'] = [[list(row) for row in cov] for cov in covs]
    output['data']['weights'] = list(weights)
    output['data']['noise'] = data['noise']*(np.array(f_Vs).max()-np.array(f_Vs).min())

    opt_fns = {'min': np.nanargmin, 'max': np.nanargmax}
    for opt in ['min','max']:    
        output['optimum'][opt] = {}
        argind = argopt(omega_Vs,opt_fns[opt])
        output['optimum'][opt]['arg'] = list(omega_Ps[argind[0]][argind[1]])
        output['optimum'][opt]['value'] = omega_Vs[argind[0]][argind[1]]

    output['input_file_name'] = meta_name # meta_input['input_file_name'] 
    output['objective_id'] = objective_id
    generated_file = ''.join(['_'.join([output_prefix,objective_id]),'.',output_suffix])
    output['generated_file_name'] = generated_file

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info('Finished output for objective ID {}.'.format(objective_id))

    return output, generated_file

def main(data_root: str, args: dict, logger_name: Optional[str]=None,is_raised: Optional[bool]=True):

    with open(os.path.join(data_root,'inputs',args.input_file),'r') as file:
        meta_input = json.load(file)
        meta_name = args.input_file

    legit_input(meta_input,logger_name=logger_name,is_raised=is_raised)

    n = calculate_objectives(meta_input,args)

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info('Producing %s objectives', n)

    for _ in range(n):

        try:

            obj_output, obj_file = generate_objective(meta_input,meta_name,logger_name=logger_name,is_raised=is_raised)
            obj_path = os.path.join(data_root,'objectives',obj_file)

            with open(obj_path,'w') as file:
                json.dump(obj_output, file)

        except TypeError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Failed to dump generated objective into json.\n')
                log.error(e)
            if is_raised: raise e
        except Exception as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Something went wrong while processing objective...\n')
                log.error(e)
            if is_raised: raise e

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Specify name of input json file from data inputs dir.")
    parser.add_argument("-o", "--objectives", type=int, default=1, help="Number of simulation objectives to produce.")
    parser.add_argument("-l", "--logger", action="store_true", help="Enable logging.")
    args = parser.parse_args()

    configs_path = os.environ['HOME']
    configs_file = 'configs.yaml'

    with open(os.path.join(configs_path,configs_file),'r') as file:
        try:
            configs = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print('CRITICAL ... Unable to load configs yaml.\n.')
            print(e)
            raise e    

    user = getpass.getuser()
    data_root = configs[user]['data']

    now = datetime.now().strftime('%Y-%m-%d_%H%M')
    log_file = 'generanted_objective_{}.log'.format(now)
    log_path = os.path.join(data_root,'logs',log_file)
    logger_name = 'generanted_objective_log' if args.logger else None
    setup_logger(logger_name, log_path)

    if logger_name:                
        log = logging.getLogger(logger_name)                
        log.info('Running on user %s', user)
        log.info('Data root %s', data_root)
        log.info('Parsing input file %s', args.input_file)

    main(data_root, args, logger_name)