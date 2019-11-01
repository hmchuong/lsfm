from setuptools import setup, find_packages
from os.path import join
import versioneer

install_requires = ['joblib>=0.9.4',
                    'docopt>=0.6.2']

setup(name='lsfm',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Large Scale Facial Model (LSFM) construction pipeline',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      packages=find_packages(),
      package_data={'lsfm': ['data/*']},
      scripts=[join('bin', 'lsfm')],
      install_requires=install_requires
      )
