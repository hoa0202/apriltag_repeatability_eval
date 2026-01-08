from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'apriltag_repeatability_eval'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='AprilTag 기반 경로 반복성(Repeatability) 평가 패키지',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'collect_edges = apriltag_repeatability_eval.nodes.collect_edges:main',
            'localize_and_record = apriltag_repeatability_eval.nodes.localize_and_record:main',
        ],
    },
)
