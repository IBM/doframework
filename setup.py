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

from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'ray[default,serve,k8s]>=1.4.1',
    'rayvens>=0.5.0',
    'ibm-cos-sdk>=2.10.0',
    'boto3>=1.17.110',
    'aiohttp>=3.7.4',
    'aioredis>=1.3.1',
    'scikit-learn>=0.24.1',
    'scipy',
    'PuLP>=2.4',
    'GPy>=1.9.9'
]

setup(name='doframework',
version='0.1.3',
description='A testing framework for data-driven decision-optimization algorithms.',
long_description_content_type="text/markdown",
long_description=open('README.md').read(),
author='Orit Davidovich',
author_email='orit.davidovich@ibm.com',
url="https://github.com/IBM/doframework",
license="Apache-2.0",
classifiers=[
    "Programming Language :: Python :: 3",
    'Programming Language :: Python :: 3 :: Only',
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent"
    ],
packages=find_packages(include=['doframework', 'doframework.core', 'doframework.flow']),
install_requires=install_requires,
scripts=['doframework-setup.sh'],
package_data={'doframework': ['notebooks/*.ipynb', 'inputs/*.json', 'outputs/*.json', 'outputs/*.csv', 'examples/*.py']},
python_requires='>=3.8'
)
