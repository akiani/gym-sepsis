from setuptools import setup

setup(name='gym_sepsis',
      version='0.0.2',
      install_requires=['gym'],
      package_data={
          'model': ['model/sepsis_mortality.model',
                    'model/sepsis_starting_states.npz',
                    'model/sepsis_states.model']
      }
      )
