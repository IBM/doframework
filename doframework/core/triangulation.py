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

from dataclasses import dataclass
from functools import partial
import itertools as it

from math import factorial
import numpy as np
from numpy import linalg as la
from numpy.random import choice
from scipy.stats import rv_continuous, beta, uniform, bernoulli
from scipy.spatial import ConvexHull

from typing import List, Union, Optional
import logging

from doframework.core.pwl import Polyhedron, PolyLinear
from doframework.core.utils import flatten, logger, minimum_stats, sample_standard_simplex
from doframework.core.sampler import sample_f_values

##################################################################################################################
##################################################################################################################

@dataclass
class Process:
    '''
    Class for sampling processes.
    '''
        
    hypothesis: rv_continuous
        
    def sample(self, N: int=1) -> np.array:
        
        return self.hypothesis.rvs(N)

##################################################################################################################
##################################################################################################################

@dataclass
class Node:
    '''
    Class for node of a tree with polyhedron data.
    '''

    poly: Union[Polyhedron,PolyLinear]
    children: List[Union[Polyhedron,PolyLinear]] = None

    def add_child(self, child):
        if self.children is not None:
            self.children.append(child)
        else:
            self.children = [child]

##################################################################################################################
##################################################################################################################

@dataclass
class Tree:
    '''
    Class of polyhedra trees.
    '''
    
    root: Node

    def find(self, points: np.array=None) -> Optional[Node]:
        '''
        Find the lowest node that contains ALL points. This assumes a single lowest node, namely, all points are NOT on the codim 1 skeleton.
        It will NOT be sufficient when all points are on a single codim 1 skeleton facet which belongs to two or more polyhedra.
        '''
    
        tree_node = None
        
        if (points is not None) and (points.shape[0]):

            queue = [self.root]

            while queue:

                queue_node = queue.pop(0)

                if queue_node.poly.isin(points).all():

                    tree_node = queue_node
                    
                    if queue_node.children is not None:
                        
                        queue = [child for child in queue_node.children] # change queue w/o changing node children on pop

        return tree_node
    
    @logger
    def leaves(self, points: np.array=None, is_contained: bool=True, 
               logger_name: Optional[str]=None, is_raised: Optional[bool]=False) -> List[Node]:
        '''
        Find leaves containing points or contained in hull of points. 
        Input points are assumed inside Hull(tree). If points=None, this produces all tree leaves. 
        Leaves of polyhedra that neither contain points nor are in hull of points will be excluded.
        '''

        leaves = []

        if points is None:

            points = self.root.poly.points

        queue = [self.root]

        if is_contained:

            if points.shape[0] <= points.shape[-1]:
                
                if logger_name:
                    log = logging.getLogger(logger_name)        
                    log.warning('Not enough points for is_contained=True. Received %d. There should be >= %d.' % (points.shape[0],points.shape[-1]+1))
                
                return leaves

            poly = Polyhedron(points)
            
        else:
            
            if points.shape[0] == 0:

                if logger_name:
                    log = logging.getLogger(logger_name)        
                    log.warning('Received an empty array in is_contained=False mode.')                
                
                return leaves
            
        while queue:

            queue_node = queue.pop(0)

            node_in_points = poly.isin(queue_node.poly.points).all() if is_contained else False
            points_in_node = queue_node.poly.isin(points).all()

            if node_in_points or points_in_node: 

            #### NOTE: most general logic should be intersection.
            ####       but intersection is a constrained opt problem.
            ####       we intend to keep the problem combinatorial.                

                if queue_node.children is None:                   
                    if is_contained and node_in_points:
                        leaves.append(queue_node)
                    elif (not is_contained) and points_in_node:
                        leaves.append(queue_node)
                    else:
                        pass
                else:
                    queue = queue + [child for child in queue_node.children] # change queue w/o changing node children on pop

        return leaves

##################################################################################################################
##################################################################################################################

@logger
def regularization(Ps: List[Union[Polyhedron,PolyLinear]],
                             min_probability: float,
                             logger_name: Optional[str]=None, 
                             is_raised: Optional[bool]=False) -> bool:
    if len(Ps) == 0:

        if logger_name:
            log = logging.getLogger(logger_name)        
            log.warning('Regularization received empty list. Returned False.')

        return False

    else:

        dims = [P.dim for P in Ps]
        assert min(dims) == max(dims), 'All input polyhedra must have the same dimension.'
        dim = min(dims)

        volumes = [P.volume() for P in Ps]

        return minimum_stats(volumes) > min_probability

##################################################################################################################
##################################################################################################################

@logger
def init_regularization(P: np.array, R_vol: float, min_probability: float, 
                        logger_name: Optional[str]=None, is_raised: Optional[bool]=False) -> bool:
    '''
    Check regularization for initial simplex. 
    
            Parameters:
                    P (np.array): simplex (shape=(dim+1,dim)). 
                    R_vol (float): volume of box range where P was sampled.
                    min_probability (float): probability for regularization condition. In (0,1).
                    
            Returns:
                    True if initial regularization condition holds, False otherwise.
    '''
    
    assert P.shape[0]-1==P.shape[-1], "To check initial regularization condition, array P must be of shape (dim+1,dim)."

    dim = P.shape[-1]
    simplices = [np.delete(P,i,axis=0) for i in range(P.shape[0])]    
    simplex_vols = [simplex_volume(simplex) for simplex in simplices]  
    min_volume_ratio = 1/R_vol * 1/factorial(dim) # the volume of a corner of a [0,1]^dim cube in R

    return (minimum_stats(simplex_vols) > min_probability) and (simplex_volume(P) / R_vol > min_volume_ratio)

##################################################################################################################
##################################################################################################################

@logger
def is_span(point: np.array, 
            vectors: np.array, 
            tolerance: float=1e-12,
            logger_name: Optional[str]=None, 
            is_raised: Optional[bool]=False) -> bool:
    '''
    Find if point is in the co-dim 1 hyperplane passing through vectors. 
    
            Parameters:
                    vectors (np.array): vectors that span a subspace. 
                    point (np.array): point in space.
                    
            Returns:
                    True if point is in the span of vectors, False otherwise.
    '''
    
    point = np.atleast_2d(point)
    assert vectors.shape[0] == point.shape[-1], \
    'Number of vectors must equal the dimension of point. Received %d vectors. Point dimension is %d.' % (vectors.shape[0],point.shape[-1])
                        
    return np.isclose(la.det(vectors-point),0,tolerance)

##################################################################################################################
##################################################################################################################

is_span_vect = np.vectorize(is_span,signature='(n),(n,n)->()') 

##################################################################################################################
##################################################################################################################

# arr = np.array([[1,2],[2,3],[2,3],[4,5]])
# row = np.array([[4,5],[2,3]])
# compare_vect(arr,row)
# >>> array([[False, False, False,  True],
# ...        [False,  True,  True, False]])

# TODO: change '==' to np.close to 0

compare_vect = np.vectorize(lambda arr, row: partial(np.all, axis=1)(arr==row),signature='(n,m),(m)->(n)')

##################################################################################################################
##################################################################################################################

def visible_facets(points: np.array,point: np.array) -> np.array:
    '''
    Produce the facets of the convex hull of points visible from point.
    
            Parameters:
                    points (np.array): array of points producing a convex hull.
                    point (np.array): a single point.
                    
            Returns:
                    Simplices of the triangulation.
                    
    '''    

    hull = ConvexHull(np.vstack([np.atleast_2d(point),points]),qhull_options='QG0')

    return hull.points[hull.simplices[hull.good],:]

##################################################################################################################
##################################################################################################################

def extend_affine(point: np.array, poly: Union[Polyhedron,PolyLinear]) -> np.array:
    
    return np.array([poly.coeffs @ np.pad(point.flatten(),(0,1),constant_values=1)])

##################################################################################################################
##################################################################################################################

@logger
def simplex_volume(simplex: np.array, 
                   logger_name: Optional[str]=None, is_raised: Optional[bool]=False) -> List[float]:
    '''
    Calculate the volume of input simplex. Use the fact that the volume of the k-dimensional parallelepiped 
    spanned by x_1,...,x_k equals the sqrt of the Gram matrix determinant. 
    
            Parameters:
                    simplex (np.array): simplex as a numpy arrays.
                    
            Returns:
                    volume.
                    
    '''    

    d = simplex.shape[0]
    assert d >= 1, 'Given simplex is empty.'

    shift = simplex[:-1,:]-np.repeat(simplex[-1:,:],simplex.shape[0]-1,axis=0)

    return np.sqrt(la.det(shift @ shift.T)) / factorial(d-1)

##################################################################################################################
##################################################################################################################

@logger
def sample_skeleton(trees: Union[Tree,List[Tree]], logger_name: Optional[str]=None, is_raised: Optional[bool]=False) -> np.array:
    '''
    Sample a single point on the codim 1 skeleton of given trees.
    
            Parameters:
                    trees (Union[Tree, List[Tree]]): tree or list of trees to sample from.
                    
            Returns:
                    single vector.
                    
    '''    
    
    if isinstance(trees,Tree): trees = [trees]
        
    dims = [tree.root.poly.dim for tree in trees]
    assert min(dims) == max(dims), 'Something went wrong ... All trees must have the same dimension.'
    d = min(dims)

    leaves = flatten([[leaf.poly.points for leaf in tree.leaves()] for tree in trees])
    skeleton = flatten([[np.delete(leaf,i,axis=0) for i in range(leaf.shape[0])] for leaf in leaves])
    skeleton_flat = np.unique(np.vstack([simplex.flatten() for simplex in skeleton]),axis=0)
    skeleton_unique = [simplex.reshape(d,d) for simplex in skeleton_flat]
    skeleton_vols = [simplex_volume(simplex) for simplex in skeleton_unique]

    weights = skeleton_vols / sum(skeleton_vols)
    facet = skeleton_unique[choice(len(skeleton_unique), p=weights.flatten())]
    convex_coeffs = sample_standard_simplex(facet.shape[-1])
    
    return (convex_coeffs @ facet)[None,:]

##################################################################################################################
##################################################################################################################

def outside_supp(trees: List, v: np.array, f_range: List[float], regularization_min_probability: float, q: Process,
                    logger_name: Optional[str]=None,is_raised: Optional[bool]=False) -> bool:
    '''
    Integrate new vertex v outside Hull(trees) into trees.
    
            Parameters:
                    trees (List): list of trees or lists of trees.
                    v (np.array): a single vertex (1 dimensional).
                    f_range (List): range of function values to draw from.
                    regularization_min_probability: minimum probability for the regularization condition.
                    q (Process): process to sample for bernoulli trial.
                    
            Returns:
                    True when integration was successful, False otherwise.
                    
    '''    
    
    if logger_name:
        log = logging.getLogger(logger_name)        

    trees_updated = False    
    trees_flat = flatten(trees)

    v_facets = visible_facets(np.vstack([tree.root.poly.points for tree in trees_flat]),v)
    v_polys = []

    for facet in v_facets:
        for tree in trees_flat:
            node = tree.find(facet)
            if node is not None:
                v_polys.append((facet,node))
    
    v_leaves = [(facet,Tree(node).leaves()) for facet,node in v_polys]
    v_facets = [] # update to account for degeneracy

    for facet, leaves in v_leaves:    
        
        if logger_name:
            log.debug('Facet facing vertex: %s' % [list(row) for row in facet])
        
        for leaf in leaves:

            # print('leaf:',[row for row in leaf.poly.points])

            simplices = [np.delete(leaf.poly.points,i,axis=0) for i in range(leaf.poly.points.shape[0])]    
            incidents = [i for i, simplex in enumerate(simplices) if is_span_vect(simplex,facet).all()]

            # print('incidents:',incidents)

            if len(incidents) == 1:
                v_facets.append((leaf.poly,incidents[0]))

    try: 
        v_new_polys = [(poly,
                        facet_index,
                        PolyLinear(np.vstack([np.delete(poly.points,facet_index,axis=0),v]),
                                np.zeros(facet.shape[0]+1))) for poly,facet_index in v_facets]
    except:
        v_new_polys = [] 
        if logger_name:
            log.warning('Unable to establish new polyhedra, perhaps due to QHull failure. Will return tree_updated=False.')             

    #### TODO: improve regularization when there's a single facet and a single new poly
    ####       perhaps add the "old" poly on the other side of the facet ???
    if regularization([new for _, _, new in v_new_polys], regularization_min_probability):

        toss = bernoulli(q.sample()).rvs()

        v_val = extend_affine(v,v_new_polys[choice(len(v_new_polys))][0]) if toss else sample_f_values(f_range)

        for old_poly, facet_index, new_poly in v_new_polys:
            new_poly.set_values(np.concatenate([np.delete(old_poly.values,facet_index),v_val])) # arrays assumed flat

        trees.append([Tree(Node(new)) for _, _, new in v_new_polys])
        trees_updated = True
        
    return trees_updated

##################################################################################################################
##################################################################################################################

def inside_supp(tree: Tree, v: np.array, f_range: List, regularization_min_probability: float, q: Process,
                logger_name: Optional[str]=None,is_raised: Optional[bool]=False) -> bool:
    '''
    Integrate new vertex v within Hull(tree) into tree.
    
            Parameters:
                    tree (Tree): tree object.
                    v (np.array): a single vertex (1 dimensional).
                    f_range (List): range of function values to draw from.
                    regularization_min_probability: minimum probability for the regularization condition.
                    q (Process): process to sample for bernoulli trial.
                    
            Returns:
                    True when integration was successful, False otherwise.
                    
    '''    
    if logger_name:
        log = logging.getLogger(logger_name)        
    
    tree_updated = False
    
    v_node = tree.find(v)

    try: 
        v_new_polys = [PolyLinear(np.vstack([np.delete(v_node.poly.points,i,axis=0),v]),
                                    np.zeros(v_node.poly.points.shape[0])) for i in range(v_node.poly.points.shape[0])]
    except:
        v_new_polys = [] 
        if logger_name:
            log.warning('Unable to establish new polyhedra, perhaps due to QHull failure. Will return tree_updated=False.')             


    if regularization(v_new_polys, regularization_min_probability):

        toss = bernoulli(q.sample()).rvs()
        v_val = extend_affine(v,v_node.poly) if toss else sample_f_values(f_range)

        for i, new_poly in enumerate(v_new_polys):
            new_poly.set_values(np.concatenate([np.delete(v_node.poly.values,i),v_val])) # value arrays assumed flat
            v_node.add_child(Node(new_poly))        
        
        tree_updated = True
        
    return tree_updated

##################################################################################################################
##################################################################################################################

def skeleton_supp(trees: List[Tree], v: np.array, f_range: List, regularization_min_probability: float,
                  logger_name: Optional[str]=None, is_raised: Optional[bool]=False) -> bool:
    '''
    Integrate new vertex v on Hull(trees) skeleton into trees.
    
            Parameters:
                    trees (List): list of trees.
                    v (np.array): a single vertex (1 dimensional).
                    f_range (List): range of function values to draw from.
                    regularization_min_probability: minimum probability for the regularization condition.
                    
            Returns:
                    True when integration was successful, False otherwise.
                    
    '''    
    
    trees_updated = False    
    is_regular = True

    if logger_name:
        log = logging.getLogger(logger_name)        

    v_new_polys = []
    v_nodes = []

    for tree in trees:
        v_nodes += tree.leaves(v,is_contained=False,logger_name=logger_name, is_raised=is_raised)
        
    if len(v_nodes) in [1,2]:

        for v_node in v_nodes:
            
            simplices = [np.delete(v_node.poly.points,i,axis=0) for i in range(v_node.poly.points.shape[0])]    
            incidents = [i for i, simplex in enumerate(simplices) if is_span(v,simplex,logger_name=logger_name, is_raised=is_raised).all()]

            if len(incidents) == 1:
                
                try:                   
                    v_index = incidents[0]
                    poly_facet = np.delete(v_node.poly.points,v_index,axis=0)
                    v_new_polys += [(v_node,[PolyLinear(np.vstack([np.delete(poly_facet,j,axis=0),v_node.poly.points[v_index,:],v]),
                                                        np.zeros(poly_facet.shape[0]+1)) for j in range(poly_facet.shape[0])])]
                except:
                    if logger_name:
                        log.warning('Unable to establish new polyhedra, perhaps due to QHull failure. Will return tree_updated=False.')             
                
            else:
                
                if logger_name:
                    log.warning('Given vertex %s should be in exactly one facet of %s.' % ([list(row) for row in v], [list(row) for row in v_node.poly.points]))
                                                    
        for _, new_polys in v_new_polys:

            is_regular *= regularization(new_polys, regularization_min_probability)
            
        if len(v_new_polys) > 0:

            if is_regular:

                v_val = sample_f_values(f_range)

                for v_node, new_polys in v_new_polys:

                    for new_poly in new_polys:

                        poly_vals = compare_vect(v_node.poly.points, new_poly.points[:-1]) @ v_node.poly.values
                        new_poly.set_values(np.concatenate([poly_vals, v_val])) # value arrays assumed flat
                        v_node.add_child(Node(new_poly))

                trees_updated = True
                
        else:
            if logger_name:
                log.warning('Given vertex %s should have produced new polyhedra. Tree was not updated.' % [list(row) for row in v])
                        
    else:
        if logger_name:
            log.warning('Given vertex %s should either be in one or two tree polyhedra. Tree was not updated.' % [list(row) for row in v])
                
    return trees_updated


##################################################################################################################
##################################################################################################################

@logger
def supp_volume(polys: List,
                logger_name: Optional[str]=None, is_raised: Optional[bool]=False) -> float:

    total_volume = 0

    for T in polys:

        if isinstance(T,Tree):
            total_volume += T.root.poly.volume()
        elif isinstance(T,list):
            for tree in T:
                total_volume += tree.root.poly.volume()
        else:
            assert True, 'Item is neither a Tree nor a list of Trees'
            
    return total_volume


##################################################################################################################
##################################################################################################################

@logger
def get_omega(polys: List, total_volume: float, ratio: float,
                logger_name: Optional[str]=None, is_raised: Optional[bool]=False) -> List:
    
    accum_volume = 0
    omega = []
    i = 0

    while accum_volume/total_volume < ratio:

        T = polys[i]

        if isinstance(T,Tree):
            accum_volume += T.root.poly.volume()
            omega += [T]
            i += 1
        elif isinstance(T,list):
            for tree in T:
                accum_volume += tree.root.poly.volume()
            omega += [T]
            i += 1        
        else:
            assert True, 'Item is neither a Tree nor a list of Trees'

    if len(omega) == 0: 

        T = polys[0]        
        omega = T if isinstance(T,Tree) else T[0] #### TODO: look for a leaf of T that satisfies volume ratio???

    return omega


##################################################################################################################
##################################################################################################################

def pts_iterator(P: Union[Polyhedron,PolyLinear], N: int=1):
    while True:
        yield P.sample(N)

def box_iterator(R,N):
    while True:
        yield np.array([uniform.rvs(*r,N) for r in R]).T

##################################################################################################################
##################################################################################################################

def triangulation(R: np.array, f_range: List[float], ratio: float, N: int, d: int,                   
                  p: Process, q: Process, regularization_min_probability: float, M: int=10,
                  logger_name: Optional[str]=None, is_raised: Optional[bool]=False) -> List[Union[Tree,List[Tree]]]:
    '''
    Sample a vertex set within the given polyhedron R and generate a triangulation for that vertes set. 
    Sample a value per vertex to produce a PWL function f. Produce a fesibility region Omega within the support of f,
    with ratio as the lower bound on its volume relative to the volume of the support of f.
    
            Parameters:
                    R (np.array): A box range where triangulation vertices will be sampled. 
                    f_range (List[float]): a range of PWL function values, one for each coordinate.
                    ratio (float): a lower bound on vol(Omega)/vol(Supp(f)).
                    N (int): number of vertices to sample.
                    d (int): problem dimension [i.e., the dimension of R].
                    p (Process): sampling object for interior/exterior or codimension 1 skeleton sampling decision.
                    q (Process): sampling object for function value generation decision.
                    regularization_min_probability (float): the minimal probability for the regularization condition.
                    M (int): number of repreated regularization attempts per sampled vertex.
                    
            Returns:
                    Returns a list of trees that perserve the DAG of PWL polyhedra as nodes.
    '''

    if logger_name:
        log = logging.getLogger(logger_name)        

    R_vol = np.prod(np.vstack(R).max(axis=1)-np.vstack(R).min(axis=1))
    init_pts = list(it.islice(filter(lambda P: init_regularization(P, R_vol, regularization_min_probability,logger_name=logger_name,is_raised=is_raised),box_iterator(R,d+1)),1))[0]
    init_poly = PolyLinear(init_pts,sample_f_values(f_range,init_pts.shape[0]))

    f_support = []
    f_support.append(Tree(Node(init_poly)))

    i = 0
    j = 0

    while (i <= N - (d+1)) and (j <= M*(N - (d+1))):

        j += 1

        toss = bernoulli(p.sample()).rvs()

        if toss:

            trees = flatten(f_support)
            v = sample_skeleton(trees,logger_name=logger_name, is_raised=is_raised)

            if v is not None:            

                if skeleton_supp(trees, v, f_range, regularization_min_probability, logger_name=logger_name, is_raised=is_raised):

                    if logger_name:
                        log.debug('Vertex on skeleton: %s' % [list(row) for row in v])

                    i += 1

                else:

                    continue

            else:

                if logger_name:
                    log.warning('Failed to sample vertex on skeleton on the {}-th iteration. This could be due to a numerical problem in isin.'.format(j))

                j -= 1

        else: 

            v = next(box_iterator(R,1)) # R.sample()

            try:

                v_tree = next(tree for tree in flatten(f_support) if tree.root.poly.isin(v))

                if inside_supp(v_tree, v, f_range, regularization_min_probability, q, logger_name-logger_name, is_raised=is_raised):

                    if logger_name:
                        log.debug('Vertex inside: %s' % [list(row) for row in v])

                    i += 1

                else:

                    continue

            except: 

                if outside_supp(f_support, v, f_range, regularization_min_probability, q, logger_name=logger_name, is_raised=is_raised): 

                    if logger_name:
                        log.debug('Vertex outside: %s' % [list(row) for row in v])

                    i += 1

                else:

                    continue

    f_volume = supp_volume(f_support)
    omega = get_omega(f_support, f_volume, ratio)

    return f_support, omega