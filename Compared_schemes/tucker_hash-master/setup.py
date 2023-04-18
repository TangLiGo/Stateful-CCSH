from setuptools import setup, find_packages


install_requires=[
   'ImageHash>=4.0',
   'numpy>=1.17.4',
   'Pillow>=6.2.1',
   'scikit-image>=0.16.2',
   'scipy>=1.3.3',
   'tensorly>=0.4.5'
]


setup(
    name='tuckerhash',
    version='0.0.1',
    description='Tucker decomposition for image hashing',
    url='https://github.com/martins0n/nla_project_image_hash',
    packages=find_packages(exclude=('tests', 'notebooks')),
    install_requires=install_requires
)