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
from doframework.core.triangulation import Process, triangulation, box_iterator
from doframework.core.utils import flatten, sample_standard_simplex, value_match, incidence
from doframework.core.pwl import PWL, Polyhedron, argopt
from doframework.core.hit_and_run import get_hull, in_domain, scale, hit_and_run

def calculate_objectives(meta_input: dict, args: dict) -> int:
    f = meta_input['f']
    omega = meta_input['omega']
    objectives = 1 if ('position' in f['vertices']) and \
                    ('vertices' in omega) and \
                    ('position' in omega['vertices']) and \
                    (('evals' in f['values']) or ('coeffs' in f['values'])) else args.objectives
    return objectives
        
def generate_objective(meta_input: dict, meta_name: str, **kwargs) -> Tuple[dict, str]:
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

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    tol = kwargs['tol'] if 'tol' in kwargs else 1e-8

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

    f_vertex_num = parse_vertex_num(meta_input)
    fpoints, opoints = np.array([]), np.array([])

    assert f_vertex_num > 0, 'Vertex number for PWL function f not specified. Provide f[\"vertices\"][\"num\"].'

    if 'position' in f['vertices']:
        
        fpoints = np.atleast_2d(np.array(f['vertices']['position']))
        fhull, fvertices, fsimplices = get_hull(fpoints)
        
        assert 'vertices' in omega,\
        'When specifying \"position\" for f vertices, omega vertices must also be specified. You can either add \"num\" and \"range\" fields under omega[\"vertices\"] for num vertices to be sampled within range or choose to specify omega vertices directly in \"position\" field under omega[\"vertices\"].'
        
    else:
        
        assert 'range' in f['vertices'], 'f[\"vertices\"] is missing \"range\" field to sample from.'
        
        f_domain_range = np.atleast_2d(np.array(f['vertices']['range']))
        f_domain_vertices = np.vstack(list(map(np.array, it.product(*f['vertices']['range']))))
        fhull, fvertices, fsimplices = get_hull(f_domain_vertices)
        

    if 'vertices' in omega:
        
        if 'position' in omega['vertices']:
        
            opoints = np.atleast_2d(np.array(omega['vertices']['position']))
            opoints = opoints[in_domain(opoints, fhull.equations, tol=tol)]
            if fpoints.size == 0:
                fpoints = f_domain_vertices
            
        else:
            
            assert 'num' in omega['vertices'], 'omega[\"vertices\"] is missing \"num\" field for number of omega points sampled.'
            
            omega_vertex_num = omega['vertices']['num']
            
            if 'range' in omega['vertices']:
                
                if fpoints.size == 0:
                    
                    fpoints = f_domain_vertices
                    
                opoints = np.vstack(
                    list(
                        it.islice(
                            filter(lambda point: in_domain(np.atleast_2d(point), fhull.equations, tol=tol)[0],
                                box_iterator(omega['vertices']['range'],1)),
                            omega_vertex_num)
                    )
                )
                
            else:
                
                if fpoints.size > 0:
            
                    cm = np.average(fvertices,axis=0)[None,:]
                    ratio = scale(cm, fsimplices[:,0,:], fhull.equations[:,:-1])

                    tmp_points = (fpoints-cm)/ratio 
                    tmp_hull, tmp_vertices, tmp_simplices = get_hull(tmp_points)
                    origin = np.average(tmp_vertices,axis=0)[None,:]

                    tmp_opoints = hit_and_run(omega_vertex_num,origin,tmp_hull.equations,T=1,delta=0.1,num_cpus=1,tol=tol)
                    opoints = tmp_opoints*ratio+cm
                    
    else:
        
        if fpoints.size>0 and omega['ratio']>=1:
            opoints = fpoints

    if fpoints.size>0 and opoints.size>0:    
        
        if 'coeffs' in f['values']:
        
            fcoeffs = f['values']['coeffs']
            fvals = np.pad(fpoints,[(0,0),(0,1)],constant_values=1) @ fcoeffs
            ovals = np.pad(opoints,[(0,0),(0,1)],constant_values=1) @ fcoeffs

        else:    

            if 'evals' in f['values']:

                assert 'position' in f['vertices'], \
                'There must be a \"positions\" field in f[\"vertices\"], when \"evals\" field is specified in f[\"values\"].'
                assert len(f['values']['evals']) == len(f['vertices']['position']),\
                'The length of teh values list under f[\"values\"][\"evals\"] should match the number of \"positions\" in f[\"vertices\"].'

                fvals = np.array(f['values']['evals'])
                f_value_range = [fvals.min(),fvals.max()]
                ovals = value_match(fpoints,opoints,fvals,f_value_range)

            else:

                assert 'range' in f['values'], \
                'There must be a \"range\" field in f[\"values\"], when neither \"evals\" nor \"coeffs\" are specified.'

                f_value_range = f['values']['range'] 
                fvals = uniform.rvs(f_value_range[0],f_value_range[1]-f_value_range[0],fpoints.shape[0])
                ovals = uniform.rvs(f_value_range[0],f_value_range[1]-f_value_range[0],opoints.shape[0])

        m = fpoints.min()
        M = fpoints.max()

        olift = np.hstack([opoints,(np.random.rand(opoints.shape[0])*(M-m)+m)[:,None]])
        flift = np.hstack([fpoints,(np.random.rand(fpoints.shape[0])*(M-m)+11*(M-m)+m)[:,None]]) # fpoints far above opoints to get hull of opoints in triangulation

        P = np.vstack([opoints,fpoints])
        _, unique_indices = np.unique(P, axis=0, return_index=True) # first unique occurance in omega
        Plift = np.vstack([olift,flift])[unique_indices]

        view_point = np.concatenate([P.mean(axis=0),np.array([m-1000*(M-m)])]) # view point far below to catch full lower envelope
        envelope = ConvexHull(np.vstack([np.atleast_2d(view_point),Plift]),qhull_options='QG0')
        good_indices = envelope.simplices[envelope.good]
        fPs = envelope.points[good_indices,:][:,:,:-1]

        V = np.concatenate([ovals,fvals])[unique_indices]
        fVs = V[:,None][good_indices-1].reshape(*good_indices.shape) # view point at index 0

        oin = [np.all(incidence(opoints,fp).any(axis=0)) for fp in fPs]
        oPs = fPs[oin]
        oVs = fVs[oin]
        
        if oPs.size == 0: # when fail to catch omega lower envelope
            oPs, oVs = fPs, fVs
            
    else:
        
        assert 'ratio' in omega, 'field ratio is missing under omega.'
        assert 'range' in f['values'], 'field \"range\"" (for range of function values) is missing under f[\"values\"].'
        
        ratio = omega['ratio']
        f_value_range = f['values']['range']
        f_vertex_num = parse_vertex_num(meta_input)

        p = kwargs['p'] if 'p' in kwargs else 0.5
        q = kwargs['q'] if 'q' in kwargs else 1

        # default prob: use upper bound on number of simplices f_vertex_num-dim
        regularization_min_prob_default = beta(1,f_vertex_num-dim-1).cdf(1/(f_vertex_num-dim) - beta(1,f_vertex_num-dim-1).std())
        regularization_min_prob = kwargs['regularization_min_prob'] if 'regularization_min_prob' in kwargs else regularization_min_prob_default

        p_process = Process(uniform(scale=p))
        q_process = Process(uniform(scale=q))

        f_supp, omega_supp = triangulation(f_domain_range, f_value_range, ratio, f_vertex_num, dim, p_process, q_process, regularization_min_prob)            
        fPs = flatten([[leaf.poly.points for leaf in tree.leaves()] for tree in flatten(f_supp)])
        fVs = flatten([[leaf.poly.values for leaf in tree.leaves()] for tree in flatten(f_supp)])
        oPs = flatten([[leaf.poly.points for leaf in tree.leaves()] for tree in flatten(omega_supp)])
        oVs = flatten([[leaf.poly.values for leaf in tree.leaves()] for tree in flatten(omega_supp)])  
  

    output['f']['polyhedrons'] = [[list(point) for point in P] for P in fPs]
    output['f']['values'] = [list(V) for V in fVs]

    pwl = PWL(fPs,fVs)
    domain_scale = np.power(pwl.volume(),1/dim) # domain "radius" parameter
    omega_scale = omega['scale'] if 'scale' in omega else 0.0

    omega_hull = ConvexHull(np.vstack(oPs))
    omega_locs = omega_hull.points[omega_hull.vertices,:] # may shift original vertices by order of 1e-8
    omega_scales = np.random.rand(*omega_locs.shape)*omega_scale*domain_scale

    output['omega']['polyhedrons'] = [[list(point) for point in P] for P in oPs]
    output['omega']['hypothesis'] = omega['hypothesis'] if 'hypothesis' in omega else 'norm'
    output['omega']['locs'] = [list(row) for row in omega_locs]
    output['omega']['scales'] = [list(row) for row in omega_scales]
    
    policies = pwl.sample(data['policy_num'])
    eigenvals = np.vstack([sample_standard_simplex(dim)*dim for _ in range(data['policy_num'])])
    corrs = [random_correlation.rvs(e) for e in eigenvals]
    sigmas = [np.random.rand(dim)*data['scale']*domain_scale for _ in range(data['policy_num'])]
    covs = [np.diag(sigma) @ corr @ np.diag(sigma) for corr, sigma in zip(corrs,sigmas)]
    weights = sample_standard_simplex(data['policy_num'])

    output['data']['N'] = data['N']
    output['data']['hypothesis'] = data['hypothesis'] if 'hypothesis' in data else 'multivariate_normal'
    output['data']['policies'] = [list(row) for row in policies]
    output['data']['covariances'] = [[list(row) for row in cov] for cov in covs]
    output['data']['weights'] = list(weights)
    output['data']['noise'] = data['noise']*(np.array(fVs).max()-np.array(fVs).min())
    output['data']['scale'] = data['scale']

    opt_fns = {'min': np.nanargmin, 'max': np.nanargmax}
    for opt in ['min','max']:    
        output['optimum'][opt] = {}
        argind = argopt(oVs,opt_fns[opt])
        output['optimum'][opt]['arg'] = list(oPs[argind[0]][argind[1]])
        output['optimum'][opt]['value'] = oVs[argind[0]][argind[1]]

    output['input_file_name'] = meta_name # meta_input['input_file_name'] 
    output['objective_id'] = objective_id
    generated_file = ''.join(['_'.join([output_prefix,objective_id]),'.',output_suffix])
    output['generated_file_name'] = generated_file

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info('Finished output for objective ID {}.'.format(objective_id))

    return output, generated_file

def main(data_root: str, args: argparse.Namespace, **kwargs):

    with open(os.path.join(data_root,'inputs',args.input_file),'r') as file:
        meta_input = json.load(file)
        meta_name = args.input_file

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    is_raised = args.is_raised

    legit_input(meta_input,logger_name=logger_name,is_raised=is_raised)

    n = calculate_objectives(meta_input,args)

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info('Producing %s objectives', n)

    for _ in range(n):

        try:

            obj_output, obj_file = generate_objective(meta_input,meta_name,logger_name=logger_name,is_raised=is_raised)
            obj_path = os.path.join(data_root,'objectives-dest',obj_file)

            with open(obj_path,'w') as file:
                json.dump(obj_output, file)

        except TypeError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
                log.error('Failed to dump generated objective into json.\n')
                log.error(e)
            if is_raised: raise e
        except AssertionError as e:
            if logger_name:
                log = logging.getLogger(logger_name)
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
    parser.add_argument("-c", "--configs", type=str, help="Specify the absolute path of the configs file.")
    parser.add_argument("-o", "--objectives", type=int, default=1, help="Number of simulation objectives to produce.")
    parser.add_argument("-l", "--logger", action="store_true", help="Enable logging.")
    parser.add_argument("-r", "--is_raised", type=bool, default=False, help="Raise assertions and terminate run.")
    args = parser.parse_args()

    configs_path = args.configs

    with open(configs_path,'r') as file:
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

    main(data_root, args, logger_name=logger_name)