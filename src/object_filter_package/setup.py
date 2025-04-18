from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'object_filter_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include message files
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rm',
    maintainer_email='rm@todo.todo',
    description='Package for filtering objects and publishing YOLO data',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_filter_node = object_filter_package.object_filter_node:main',
            'onnxrun = object_filter_package.onnxrun:main',
            'test = object_filter_package.test:main'
        ],
    },
)
