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

from dataclasses import dataclass, field
from typing import Optional
from itertools import islice
import os
import json
from pathlib import Path
import ibm_boto3
import boto3

from doframework.core.inputs import legit_configs

CamelKeysDict = {'local': "CamelFileName", 's3': "CamelAwsS3Key"}

def _get_s3_object(configs): # !!!
    
    assert 's3' in configs, 'The `_get_s3_object` function assumes `s3` sources and targets.'

    s3 = configs['s3']

    if s3['cloud_service_provider'] == 'ibm':

        return ibm_boto3.resource(service_name='s3',
                                region_name=s3['region'],
                                endpoint_url=s3['endpoint_url'],
                                aws_access_key_id=s3['aws_access_key_id'],
                                aws_secret_access_key=s3['aws_secret_access_key'])

    if s3['cloud_service_provider'] == 'aws':

        return boto3.resource(service_name='s3',
                                region_name=s3['region'],
                                aws_access_key_id=s3['aws_access_key_id'],
                                aws_secret_access_key=s3['aws_secret_access_key'])

@dataclass
class Storage:
    '''
    Class for storage object: either an S3 object or a local file system. Initializes with user configs.
    '''
        
    configs: dict
    storage_buckets: dict = field(init=False)
    missing_buckets: dict = field(init=False)
                
    def __post_init__(self):
        
        legit_configs(self.configs)
        
        if 's3' in self.configs:
            s3 = self.configs['s3']
            buckets_obj = _get_s3_object(self.configs).buckets.all()
            buckets_list = [bucket.name for bucket in buckets_obj]
            self.storage_buckets = {name: bucket for name, bucket in s3['buckets'].items() \
                                if bucket in buckets_list}
            self.missing_buckets = {name: bucket for name, bucket in s3['buckets'].items() \
                                if bucket not in buckets_list}
        if 'local' in self.configs:
            local = self.configs['local']
            self.storage_buckets = {name: bucket for name, bucket in local['buckets'].items() \
                                if Path(bucket).is_dir()}
            self.missing_buckets = {name: bucket for name, bucket in local['buckets'].items() \
                                if not Path(bucket).is_dir()}
        
    def buckets(self):
        return self.storage_buckets

    def missing(self):
        return self.missing_buckets

    def get(self,bucket,filename):        
        body_or_path = None
        
        if bucket in self.storage_buckets.values():       
            if 's3' in self.configs:
                body_or_path = _get_s3_object(self.configs).Bucket(bucket).Object(filename).get()['Body']
            if 'local' in self.configs:
                body_or_path = open(os.path.join(bucket,filename),'r')
                
        return body_or_path
    
    def put(self,bucket,content,name,content_type):
        
        assert bucket in self.storage_buckets.values(), \
        'The bucket you provided is not on the storage bucket list. Find the list by running storage.buckets().'
        assert content_type in ['json','csv'], \
        'put method uploads either jsonable (e.g., dict) or csv-esque (e.g., pd.DataFrame) content.'

        success = False
        
        try:
        
            if 's3' in self.configs:

                if content_type == 'json':
                    _get_s3_object(self.configs).Bucket(bucket).put_object(
                        Body=json.dumps(content),
                        Key=name
                    )
                if content_type == 'csv':
                    _get_s3_object(self.configs).Bucket(bucket).put_object(
                        Body=content.to_csv(index=False),
                        Key=name
                    )

                success = True

            if 'local' in self.configs:

                if content_type == 'json':                        
                    with open(os.path.join(bucket,name), "w") as path: json.dump(content, path)                            
                if content_type == 'csv':                        
                    content.to_csv(bucket,index=False)

                success = True
                        
        except Exception as e:
            
            print(e)
                
        return success
    
    def count(self,bucket,extension):
        assert bucket in self.storage_buckets.values(), \
        'The bucket you provided is not on the storage bucket list. Find the list by running storage.buckets().'
        assert extension in ['json','csv'], \
        'count method counts either json or csv files in given bucket. provide either extension=`json` or extension=`csv`.'
                
        n = None
        
        if 's3' in self.configs:

            objects = [f for f in _get_s3_object(self.configs).Bucket(bucket).objects.all() if f.key.endswith(extension)]
            n = len(objects)

        if 'local' in self.configs:

            objects = [f for f in Path(bucket).rglob('*') if Path(f).suffix in [f'.{extension}']]
            n = len(objects)

        return n

    def get_all(self,bucket,extension,limit: Optional[int]=None):        
        assert bucket in self.storage_buckets.values(), \
        'The bucket you provided is not on the storage bucket list. Find the list by running storage.buckets().'
        assert extension in ['json','csv'], \
        'get_all method retrieves either json or csv files in given bucket. provide either extension=`json` or extension=`csv`.'

        objects = []
        
        if bucket in self.storage_buckets.values():       
            if 's3' in self.configs:
                if limit:
                    objects = [f for f in _get_s3_object(self.configs).Bucket(bucket).objects.all().limit(limit) if f.key.endswith(extension)]
                else:
                    objects = [f for f in _get_s3_object(self.configs).Bucket(bucket).objects.all() if f.key.endswith(extension)]
            if 'local' in self.configs:
                if limit:
                    objects = [f for f in islice(Path(bucket).iterdir(),limit) if Path(f).suffix in [f'.{extension}']]
                else:
                    objects = [f for f in Path(bucket).rglob('*') if Path(f).suffix in [f'.{extension}']]
                
        return objects
    
