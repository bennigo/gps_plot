from setuptools import setup, find_packages

setup(name='timesmatplt',
      version='0.1',
      description='Time and date modules capable of handling GPS time',
      url='http://github.com/imo/gtimes',
      author='Benedikt G. Ofeigsson',
      author_email='bgo@vedur.is',
      license='Icelandic Met Office',
      package={'timesmatplt': 'timesmatplt', 'timesmatplt': 'gasmatplt'},
      scripts=['bin/plot-dev', 'bin/plot-gps-timeseries', 'bin/convgamit', 'bin/gps-savetimes.py','bin/plot-multigas-timeseries' ], #, 'bin/conv2png'],
      packages=find_packages(),
      zip_safe=False)
