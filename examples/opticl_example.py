import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import opticl
from pyomo import environ
from pyomo.environ import *
import doframework as dof

@dof.resolve
def predict_optimize_opticl(data: np.array, constraints: np.array, **kwargs):
    '''
    OptiCL predict-optimize algorithm.

            Parameters:
                    data (np.array): data for regression [possibly including unfeasible data points].
                    constraints (np.array): array of coefficients of equations that define the feasibility region.
                    
            Returns:
                    optimum: the location of a predicted optimal solution.
                    predicted_value: the predicted optimal value.
                    model: the regression model used for prediction.

    '''

    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    is_raised = kwargs['is_raised'] if 'is_raised' in kwargs else False
    alg_list = kwargs['alg_list'] if 'alg_list' in kwargs else ['linear','rf','svm','cart','gbm','mlp']
    task_type = kwargs['task_type'] if 'task_type' in kwargs else 'continuous'
    is_trust_region = kwargs['is_trust_region'] if 'is_trust_region' in kwargs else True
    solver = kwargs['solver'] if 'solver' in kwargs else None
    
    tolerance = 1e-8
    
    try:
        
        data_feasible = data[np.all(np.pad(data[:,:-1],((0,0),(0,1)),constant_values=1) @ constraints.T <= -tolerance, axis=1)]

        X = pd.DataFrame(data_feasible[:,:-1]).add_prefix('x')
        y = pd.DataFrame(data_feasible[:,-1:]).add_prefix('y')

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        mfiles = {}
        models = {}
        perfs = pd.DataFrame()
        response_vars = y_train.columns

        for var in response_vars:

            for alg in alg_list:

                alg_run = 'rf_shallow' if alg == 'rf' else alg

                model_path = os.path.join('results','{}'.format(alg))
                model_file = os.path.join(model_path,'{}_model'.format(var))

                m, perf = opticl.run_model(X_train, y_train[var], X_test, y_test[var], alg_run, var,
                                           task = task_type, cv_folds = 5, parameter_grid = None, 
                                           save_path = model_file, save = False)        

                constraintL = opticl.ConstraintLearning(X_train, y_train, m, alg)
                constraint_add = constraintL.constraint_extrapolation(task_type)
                mfiles[model_file] = constraint_add
                models[alg] = m

                perf['outcome'] = var
                perf['alg'] = alg
                perfs = pd.concat([perfs,perf],ignore_index=True)

        model_master = opticl.model_selection(perfs, objectives_embed = {var:1})
        model_master['lb'] = None
        model_master['ub'] = None
        model_chosen = model_master.loc[model_master.outcome == response_vars[0],'model_type'][0]
        
        model_pyo = ConcreteModel()

        N = X_train.columns
        model_pyo.x = Var(N, domain=Reals)

        for i,eqn in enumerate(constraints):
            model_pyo.add_component('constr_known{}'.format(i), Constraint(expr=sum(model_pyo.x[var]*eqn[j] for j,var in enumerate(N)) <= -eqn[-1]+tolerance))

        model_pyo.OBJ = Objective(expr=0, sense=minimize)
        
        final_model_pyo = opticl.optimization_MIP(model_pyo,
                                                  model_pyo.x,
                                                  model_master,
                                                  mfiles,
                                                  X_train,
                                                  tr=is_trust_region)
        
        opt = SolverFactory(solver)
        _ = opt.solve(final_model_pyo)
        x_solution = getattr(final_model_pyo, 'x')
        optimum = np.array([x_solution[i].value for i in N])
        predicted_value = final_model_pyo.OBJ()        
        
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

        return optimum, predicted_value, models[model_chosen]
        
if __name__ == '__main__':

    configs_path = os.getcwd()
    configs_file = 'configs.yaml'

    dof.run(predict_optimize_opticl, 
            os.path.join(configs_path,configs_file), 
            alg_list=['linear','rf','svm','cart','gbm'],
            solver='cplex', 
            objectives=1, datasets=2, after_idle_for=400, distribute=True, alg_num_cpus=4)