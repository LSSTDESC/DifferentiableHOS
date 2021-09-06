from setuptools import find_packages
from setuptools import setup

setup(name='DifferentiableHOS',
      description='Differentiable Higher Order Statistics in TensorFlow',
      author='Denise Lanzieri',
      license='MIT',
      packages=find_packages(),
      install_requires=['flowpm'],
      tests_require=['lenspack'],
      extras_require={
          'testing': ['lenspack'],
      },
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      classifiers=[
          'Development Status :: 3 - Alpha', 'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      keywords='cosmology')
