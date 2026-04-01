from setuptools import find_packages, setup

package_name = 'drone_math_model'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='rein.forcement.ligma@gmain.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'drone_model = drone_math_model.drone_model:main',
            'drone_model_test = drone_math_model.drone_tester:main',
            'drone_flat_to_state_and_torque_node = drone_math_model.drone_flat_to_state:main',
        ],
    },
)
