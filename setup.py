from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setup(
    name='smallrl',
    version='0.0.1',
    description='smallrl: personal repository for fast RL prototyping',
    author='Nicklas Hansen',
    url='https://github.com/nicklashansen/smallrl',
    author_email='hello@nicklashansen.com',
    packages=find_packages()
)
