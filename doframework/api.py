import json
from io import StringIO
from collections import namedtuple

import pandas as pd

import ray
import rayvens

from doframework.core.inputs import get_configs
from doframework.core.storage import Storage, CamelKeysDict
from doframework.flow.objectives import generate_objective, calculate_objectives
from doframework.flow.datasets import generate_dataset
from doframework.flow.solutions import generate_solution, files_from_data
from doframework.flow.metrics import generate_metric, files_from_solution
from doframework.flow.mock import generate_objective_mock, generate_dataset_mock, generate_solution_mock, generate_metric_mock, sleep_time

#################################################################################################################
#################################################################################################################

Args = namedtuple(
    'Args',[
        'objectives',
        'datasets',
        'feasibility_regions',
        'run_mode',
        'distribute',
        'mcmc',
        'logger',
        'mock',
        'after_idle_for',
        'rayvens_logs',
        'alg_num_cpus',
        'data_num_cpus'
        ]
    )

GenerateFunctionsDict = {
    'objective': generate_objective, 
    'objective_mock': generate_objective_mock, 
    'data' : generate_dataset,
    'data_mock' : generate_dataset_mock,    
    'solution' : generate_solution,
    'solution_mock' : generate_solution_mock,    
    'metric' : generate_metric,
    'metric_mock' : generate_metric_mock,    
}

#################################################################################################################
#################################################################################################################

def _get_source_config(from_bucket, to_bucket, configs):

    d = {}
    
    if 's3' in configs:

        s3 = configs['s3']

        d = dict(kind='cloud-object-storage-source',
            bucket_name=from_bucket,
            access_key_id=s3['aws_access_key_id'],
            secret_access_key=s3['aws_secret_access_key'],
            region=s3['region'],
            move_after_read=to_bucket
        )

        if 'endpoint_url' in s3:

            d = {**d,**dict(endpoint=s3['endpoint_url'])}
    
    if 'local' in configs:
        
        d = dict(kind='file-source', 
            path=from_bucket, 
            keep_file=False, 
            move_after_read=to_bucket
        )

    return d

def _get_sink_config(to_bucket, configs):

    d = {}

    if 's3' in configs:

        s3 = configs['s3']

        d = dict(kind='cloud-object-storage-sink',
            bucket_name=to_bucket,
            access_key_id=s3['aws_access_key_id'],
            secret_access_key=s3['aws_secret_access_key'],
            region=s3['region']
        )

        if 'endpoint_url' in s3:

            d = {**d,**dict(endpoint=s3['endpoint_url'])}
        
    if 'local' in configs:
        
        d = dict(kind='file-sink', 
            path=to_bucket
        )

    return d

def _target_bucket_name(buckets, process_type,args):
    try:
        if process_type == 'input':
            bucket_name = buckets['objectives'] if 'objectives' in buckets else None
        elif process_type == 'objective':
            bucket_name = buckets['data'] if 'data' in buckets else None
        elif process_type == 'data':
            bucket_name = buckets['solutions'] if 'solutions' in buckets else None
        elif process_type == 'solution':
            bucket_name = buckets['metrics_dest'] if 'metrics_dest' in buckets else None
        else:
            bucket_name = None
        return bucket_name
    except TypeError as e:
        if args.logger:
            print('({}) ERROR ... Likely received buckets=None in target_bucket_name.'.format(process_type))
            print('({}) ERROR ... '.format(process_type) + e)
    except Exception as e:
        if args.logger:
            print('({}) ERROR ... Error occured when inferring bucket name in target_bucket_name.'.format(process_type))
            print('({}) ERROR ... '.format(process_type) + e)
    return None     

def _get_event_type(process_type,args):
    try:
        if process_type in ['input','objective','solution']:
            event_type = 'json'
        elif process_type in ['data']:
            event_type = 'csv'
        else:
            event_type = None
        return event_type
    except TypeError as e:
        if args.logger:
            print('({}) ERROR ... Likely received buckets=None in target_bucket_name.'.format(process_type))
            print('({}) ERROR ... '.format(process_type) + e)
    except Exception as e:
        if args.logger:
            print('({}) ERROR ... Error occured when inferring bucket name in target_bucket_name.'.format(process_type))
            print('({}) ERROR ... '.format(process_type) + e)
    return None     

def _get_event(process_type,process_json,event_type,args):
    try:
        if event_type == 'json':
            process_input = json.loads(process_json['body'])
        elif event_type == 'csv':
            process_input = pd.read_csv(StringIO(process_json['body']))
        else:
            process_input = None                    
        return process_input
    except json.JSONDecodeError as e:
        if args.logger:
            print('({}) ERROR ... Error occured while decoding file {} in json loads.'.format(process_json['filename']))
            print('({}) ERROR ... '.format(process_type) + e)
    except Exception as e:
        if args.logger:
            print('({}) ERROR ... Error occured when extracting event content.'.format(process_type))
            print('({}) ERROR ... '.format(process_type) + e)
    return None     

def _number_of_iterations(process_input, args, process_type):
    try:
        if process_type == 'input':
            n = args.objectives if args.mock else calculate_objectives(process_input,args)
        elif process_type == 'objective':
            n = args.datasets
        elif process_type == 'data':
            n = args.feasibility_regions
        elif process_type == 'solution':
            n = 1
        else:
            n = None
        return n
    except KeyError as e:
        if args.logger:
            print('({}) ERROR ... Error occured when calculating n in number_of_iterations.'.format(process_type))
            print('({}) ERROR ... '.format(process_type) + e)
    except Exception as e:
        if args.logger:
            print('({}) ERROR ... Error occured when calculating n in number_of_iterations.'.format(process_type))
            print('({}) ERROR ... '.format(process_type) + e)
    return None     

def _get_extra_input(input_name, process_type, configs, args, buckets):
    try:
        if process_type == 'input':
            extra = {'mock': args.mock}
        elif process_type == 'objective':
            extra = {'mock': args.mock}
        elif process_type == 'data':
            files = files_from_data(input_name)
            storage = Storage(configs)
            objective = storage.get(buckets['objectives_dest'],files['objective'])
            extra = {'num_cpus': args.data_num_cpus, 'objective': json.load(objective), 'mock': args.mock}
        elif process_type == 'solution':
            files = files_from_solution(input_name)
            storage = Storage(configs)
            objective = storage.get(buckets['objectives_dest'],files['objective'])
            data = storage.get(buckets['data_dest'],files['data'])
            extra = {'is_mcmc': args.mcmc, 'objective': json.load(objective), 'data': pd.read_csv(data), 'mock': args.mock}
        else:
            extra = None
        return extra
    except Exception as e:
        if args.logger:
            print('({}) ERROR ... Error occured while getting extra input.'.format(process_type))
            print(e)
    return {}

#################################################################################################################
#################################################################################################################

def _process(process_type, configs, args, buckets, **kwargs):
    def proc(f):

        if process_type == 'data':
            @ray.remote(num_cpus=args.data_num_cpus)
            def f_dist(*args,**kwargs):
                return f(*args,**kwargs)
        elif process_type == 'solution':
            @ray.remote(num_cpus=args.alg_num_cpus)
            def f_dist(*args,**kwargs):
                return f(*args,**kwargs)
        else:
            @ray.remote(num_cpus=1)
            def f_dist(*args,**kwargs):
                return f(*args,**kwargs)

        def inner(context, event):
            
            try:
                
                process_json = json.loads(event)
                assert ('body' in process_json) and ('filename' in process_json), 'Missing fields body and / or filename in event json.'

                event_type = _get_event_type(process_type,args)
                process_input = _get_event(process_type,process_json,event_type,args)                    
                assert process_input is not None, 'Unable to extract process input. Perhaps illegal event_type={}.'.format(event_type)
                if args.logger: print('({}) INFO ... Process successfully extracted event of type {}.'.format(process_type, event_type))
                
                input_name = process_json['filename']
                assert input_name, 'Event name is None.'
                if args.logger: print('({}) INFO ... Process working on event {}.'.format(process_type, input_name))
                
                bucket_name = _target_bucket_name(buckets, process_type, args)
                assert bucket_name, 'Could not process bucket name in decorator for {}.'.format(f.__name__)
                if args.logger: print('({}) INFO ... Process working on event {} will write to bucket {}.'.format(process_type,input_name,bucket_name))
                
                n = _number_of_iterations(process_input, args, process_type)
                assert n, 'Could not process number of iterations in decorator for {}.'.format(f.__name__)
                if args.logger: print('({}) INFO ... Process will write {} products of event {} to bucket {}.'.format(process_type,n,input_name,bucket_name))

                extra = _get_extra_input(input_name, process_type, configs, args, buckets)
                assert extra is not None, 'Extra input is None for event {}.'.format(input_name)
                
                extra = {**extra,**kwargs,**configs}

                if args.distribute:
                    _ = [f_dist.remote(context, process_input, input_name, **extra) for _ in range(n)]
                else:
                    _ = [f(context, process_input, input_name, **extra) for _ in range(n)]     
                    
            except json.JSONDecodeError as e:
                if args.logger:
                    print('({}) ERROR ... Error occured while decoding {}.'.format(process_type, event_type))
                    print('({}) ERROR ... '.format(process_type))
                    print(e)
            except AssertionError as e:
                if args.logger:
                    print('({}) ERROR ... '.format(process_type))
                    print(e)
            except Exception as e:
                if args.logger:
                    print('({}) ERROR ... Unknown exception...'.format(process_type))
                    print('({}) ERROR ... '.format(process_type))
                    print(e)

        return inner    
    return proc

#################################################################################################################
#################################################################################################################

def resolve(predict_optimize):

    def inner(process_input, input_name, **kwargs):
        
        key = 'solution_mock' if 'mock' in kwargs and kwargs['mock'] else 'solution' # !!!
        return GenerateFunctionsDict[key](predict_optimize, process_input, input_name, **kwargs)

    return inner

#################################################################################################################
#################################################################################################################

def run(generate_user_solution, configs_file, **kwargs):

    objectives = int(kwargs['objectives']) if 'objectives' in kwargs else 1
    datasets = int(kwargs['datasets']) if 'datasets' in kwargs else 1
    feasibility_regions = int(kwargs['feasibility_regions']) if 'feasibility_regions' in kwargs else 1
    run_mode = kwargs['run_mode'] if 'run_mode' in kwargs else 'local'
    distribute = kwargs['distribute'] if 'distribute' in kwargs else True
    mcmc = kwargs['mcmc'] if 'mcmc' in kwargs else False
    logger = kwargs['logger'] if 'logger' in kwargs else True
    mock = kwargs['mock'] if 'mock' in kwargs else False
    after_idle_for = kwargs['after_idle_for'] if 'after_idle_for' in kwargs else 200
    rayvens_logs = kwargs['rayvens_logs'] if 'rayvens_logs' in kwargs else False
    alg_num_cpus = int(kwargs['alg_num_cpus']) if 'alg_num_cpus' in kwargs else 1
    data_num_cpus = int(kwargs['data_num_cpus']) if 'data_num_cpus' in kwargs else 1

    args = Args(objectives, datasets, feasibility_regions, run_mode, distribute, mcmc, logger, mock, after_idle_for, rayvens_logs, alg_num_cpus, data_num_cpus)

    if args.run_mode == 'operator':
        ray.init(address='auto',ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)
    rayvens.init(mode=args.run_mode ,release=(not args.rayvens_logs))

    if args.logger: print('({}) INFO ... Running simulation with args objectives={o} datasets={s} feasibility_regions={r} distribute={d} run_mode={m} logger={l}'.format('root', 
        o=args.objectives, s=args.datasets, r=args.feasibility_regions, d=args.distribute, m=args.run_mode, l=args.logger))

    if args.mock:
        if args.logger: print('({}) INFO ... Running in MOCK mode.'.format('root'))

    configs = get_configs(configs_file)
    storage = Storage(configs)
    buckets = storage.buckets()

    @ray.remote(num_cpus=1)
    @_process('input', configs, args, buckets)
    def generate_objectives(context, process_input, input_name, **kwargs):
        key = 'objective_mock' if 'mock' in kwargs and kwargs['mock'] else 'objective'    
        objective, generated_file = GenerateFunctionsDict[key](process_input, input_name, **kwargs)
        
        if any(['local' in kwargs, 's3' in kwargs]) and not all(['local' in kwargs, 's3' in kwargs]):
            key = 'local'*('local' in kwargs) + 's3'*('s3' in kwargs)
            event = rayvens.OutputEvent(json.dumps(objective),{CamelKeysDict[key]: generated_file})
            context.publish(event)
        else:
            print('({}) ERROR ... generated objective {} not published to context. Either `s3` or `local` missing from cofnigs or both feature.'.format('input',generated_file))

    @ray.remote(num_cpus=1)
    @_process('objective', configs, args, buckets)
    def generate_datasets(context, process_input, input_name, **kwargs):
        key = 'data_mock' if 'mock' in kwargs and kwargs['mock'] else 'data'    
        df, generated_file = GenerateFunctionsDict[key](process_input, input_name, **kwargs)
        
        if any(['local' in kwargs, 's3' in kwargs]) and not all(['local' in kwargs, 's3' in kwargs]):
            key = 'local'*('local' in kwargs) + 's3'*('s3' in kwargs)
            event = rayvens.OutputEvent(df.to_csv(index=False),{CamelKeysDict[key]: generated_file})
            context.publish(event)
        else:
            print('({}) ERROR ... generated dataset {} not published to context. Either `s3` or `local` missing from cofnigs or both feature.'.format('objective',generated_file))

    @ray.remote(num_cpus=1)
    @_process('data', configs, args, buckets, **kwargs)
    def generate_solutions(context, process_input, input_name, **kwargs):
        solution, generated_file = generate_user_solution(process_input, input_name, **kwargs)
        
        if any(['local' in kwargs, 's3' in kwargs]) and not all(['local' in kwargs, 's3' in kwargs]):
            key = 'local'*('local' in kwargs) + 's3'*('s3' in kwargs)
            event = rayvens.OutputEvent(json.dumps(solution),{CamelKeysDict[key]: generated_file})
            context.publish(event)
        else:
            print('({}) ERROR ... generated solution {} not published to context. Either `s3` or `local` missing from cofnigs or both feature.'.format('data',generated_file))

    @ray.remote(num_cpus=1)
    @_process('solution', configs, args, buckets)
    def generate_metrics(context, process_input, input_name, **kwargs):
        key = 'metric_mock' if 'mock' in kwargs and kwargs['mock'] else 'metric'    
        metric, generated_file = GenerateFunctionsDict[key](process_input, input_name, **kwargs)
            
        if any(['local' in kwargs, 's3' in kwargs]) and not all(['local' in kwargs, 's3' in kwargs]):
            key = 'local'*('local' in kwargs) + 's3'*('s3' in kwargs)
            event = rayvens.OutputEvent(json.dumps(metric),{CamelKeysDict[key]: generated_file})
            context.publish(event)
        else:
            print('({}) ERROR ... generated metric {} not published to context. Either `s3` or `local` missing from cofnigs or both feature.'.format('solution',generated_file))
            
    sources = ['inputs', 'objectives', 'data', 'solutions']
    targets = ['objectives', 'data', 'solutions', 'metrics_dest']
    operators = [generate_objectives, generate_datasets, generate_solutions, generate_metrics]
    streams = []

    i = (args.objectives>0) + (args.objectives>0)*(args.datasets>0) + (args.objectives>0)*(args.datasets>0)*(args.feasibility_regions>0)

    for source, target, operator in zip(sources[:i], targets[:i], operators[:i]):

        source_dest = source + '_dest'        
        source_config = _get_source_config(buckets[source], buckets[source_dest], configs)
        sink_config = _get_sink_config(buckets[target], configs)

        stream = rayvens.Stream(source)

        #### order matters: add in reverse order - sink, operator, and, lastly, source
        stream.add_sink(sink_config)
        stream.add_multitask_operator(operator)
        stream.add_source(source_config)

        streams.append(stream)

    for stream in streams:
        stream.disconnect_all(after_idle_for=args.after_idle_for)

    ray.shutdown()

#################################################################################################################
#################################################################################################################