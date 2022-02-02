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

from dataclasses import dataclass, field, InitVar
from typing import Any, Optional
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from pulp import * # LpProblem, LpMinimize, etc.

class optimalSolution:
    '''
    Class for the solutions of a decision optimization problem.
    '''
        
    def __init__(self, arg, val):
        self.arg = arg
        self.val = val
        self._post_init()
                                                        
    def _post_init(self):
        
        if self.arg is not None:
            
            if isinstance(self.arg,list):
                self.arg = np.array(self.arg)
                self.arg = self.arg.astype(float)
            if isinstance(self.arg,np.ndarray):
                self.arg = self.arg.flatten()
                self.arg = self.arg.astype(float)
            if isinstance(self.arg,float):
                self.arg = np.array([self.arg])
                self.arg = self.arg.astype(float)
                                    
        if self.val is not None:
            if isinstance(self.val,list):
                self.val = self.val[0]
                self.val = float(self.val)
            elif isinstance(self.val,np.ndarray):
                self.val = self.val.flatten()[0]
                self.val = float(self.val)
            elif isinstance(self.val,int):
                self.val = float(self.val)
                                                                
@dataclass
class predictionModel:
    '''
    Class for the prediction model of a decision optimization problem.
    '''
        
    model = LinearRegression()
    name = model.__class__.__name__
    data: InitVar[np.array] = None
    r2_score: float = field(init=False)
        
    def __post_init__(self,data):
        if data is not None:
            self.model.fit(data[:,:-1], data[:,-1])
            self.r2_score = self.model.score(data[:,:-1], data[:,-1])

    def predict(self, x: np.array) -> np.array:
        return self.model.predict(x)

@dataclass
class optimizationModel:
    '''
    Class for the solver of a decision optimization problem.
    '''

    predict: InitVar[Any]
    objective_target: np.array = field(init=False)
        
    def __post_init__(self,predict):
        self.objective_target = np.concatenate((predict.model.coef_.reshape(-1,1), 
                                                predict.model.intercept_.reshape(-1,1)))
        self.objective_target = self.objective_target.flatten()

    def optimize(self,constraints,is_minimum) -> np.array:
        
        n = self.objective_target.shape[-1]
        variables = [*range(n)]
        x = LpVariable.dicts("x", variables)

        prob = LpProblem("Optimization",LpMinimize) if is_minimum else LpProblem("Optimization",LpMaximize)
        prob += lpSum([self.objective_target[i] * x[i] for i in variables]), "Objective Target"
        prob += x[n-1] == 1, "Intercept"

        for k, eqn in enumerate(constraints):
            prob += (
                pulp.lpSum([eqn[i]*x[i] for i in variables]) <= 0,
                f"constraint_from_{k}th_facet",
            )
        prob.solve(PULP_CBC_CMD(msg=0)) # disable logs with msg=0

        return np.array([v.varValue for v in prob.variables()],dtype=np.float32)[:-1] # remove intercept coord

def predict_optimize(data: np.array, constraints: np.array, **kwargs):
    '''
    Simple predict-optimize algorithm to test package api.

            Parameters:
                    data (np.array): data for regression [possibly including unfeasible data points].
                    constraints (np.array): array of coefficients of equations that define the feasibility region.
                    is_minimum (bool): solve a min or max problem. Default is min.
                    
            Returns:
                    optimum: the location of a predicted optimal solution.
                    predicted_value: the predicted optimal value.
                    model: the regression model used for prediction.

    '''

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    is_raised = kwargs['is_raised'] if 'is_raised' in kwargs else False
    is_minimum = kwargs['is_minimum'] if 'is_minimum' in kwargs else True
                
    try:

        data_feasible = data[np.all(np.pad(data[:,:-1],((0,0),(0,1)),constant_values=1) @ constraints.T <= 0, axis=1)]
        model = predictionModel(data_feasible)
        solver = optimizationModel(model)
        optimum = solver.optimize(constraints,is_minimum)
        predicted_value = model.predict(np.atleast_2d(optimum))[0]    

    except ValueError as e:
        if logger_name:
            log = logging.getLogger(logger_name)
            log.error('ValueError: Prediction optimization failed.')
            log.error(e)
        else:
            print('ValueError: Prediction optimization failed.')
            print(e)
        if is_raised:
            raise
        else:
            return None, None, None

    except TypeError as e:
        if logger_name:
            log = logging.getLogger(logger_name)
            log.error('TypeError: Prediction optimization failed.')
            log.error(e)
        else:
            print('TypeError: Prediction optimization failed.')
            print(e)
        if is_raised:
            raise
        else:
            return None, None, None

    except Exception as e:
        if logger_name:
            log = logging.getLogger(logger_name)
            log.error('Exception: Prediction optimization failed.')
            log.error(e)
        else:
            print('Exception: Prediction optimization failed.')
            print(e)
        if is_raised:
            raise
        else:
            return None, None, None

    else:

        return optimum, predicted_value, model