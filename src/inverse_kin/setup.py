from setuptools import find_packages, setup

package_name = 'inverse_kin'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oliver',
    maintainer_email='oliver.ochoa2@iberopuebla.mx',
    description='Inverse kinematics solver and end-effector pose service server for a 6-DOF robot arm in Gazebo simulation.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'i_kin_generator = inverse_kin.i_kin_generator:main',
            'joint_state_driver = inverse_kin.joint_state_driver:main',
            'MoveToXYZServer = inverse_kin.MoveToXYZServer:main',
            'Draw_square = inverse_kin.Draw_square:main',
            'send_joint_trajectory = inverse_kin.send_joint_trajectory:main',
            'XYZServerGazebo = inverse_kin.XYZServerGazebo:main',
            'Graph_speed = inverse_kin.speed_graphing:main',
            'Trajectory_recorder = inverse_kin.Trajectory_recorder:main',
            'Trajectoryserver=inverse_kin.Trajectoryserver:main',
            "DrawPath = inverse_kin.DrawPath:main",
        ],
    },
)
