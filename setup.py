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
import pathlib

install_requires = [
    'numpy>=1.20.1',
    'pandas>=1.1.5',
    'ray>=1.4.1',
    'rayvens>=0.4.0',
    'ibm-cos-sdk>=2.10.0',
    'boto3>=1.17.110',
    'aiohttp>=3.7.4',
    'aioredis>=1.3.1',
    'scikit-learn>=0.24.1',
    'scipy>=1.7.3',
    'PuLP>=2.4',
    'GPy>=1.10.0'
]

setup(name='doframework',
version='0.1.1',
description='Testing framework for data-driven decision-optimization algorithms.',
long_description_content_type="text/markdown",
long_description=open('README.md').read(),
author='Orit Davidovich',
author_email='orit.davidovich@ibm.com',
url="https://github.com/IBM/doframework",
license="Apache-2.0",
classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent"
    ],
packages=find_packages(include=['doframework', 'doframework.core', 'doframework.flow']),
install_requires=install_requires,
package_data={'doframework': ['notebooks/*.ipynb', 'inputs/*.json']},
python_requires='>=3.9')