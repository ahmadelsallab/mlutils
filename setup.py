
import os
from setuptools import setup
from setuptools import find_packages
# requirements is generated by pipreqs <project_path>
with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
   name='mlutils',
   version='0.1',
   description='Collection of ML utilities',
   author='Ahmad El Sallab',
   author_email='ahmad.elsallab@gmail.com',
   install_requires=required, #external packages as dependencies
   packages=find_packages()
)
