from setuptools import setup, find_packages

setup(name='astrochemcd',
      description='extract molecular column density from integrated intensity maps of line emission observations',
      url='https://github.com/lunarampi',
      author='Luna Rampinelli',
      author_email='luna.rampinelli@unimi.it',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'scipy', 'astropy', 'gofish']
     )
