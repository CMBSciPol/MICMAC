from distutils.core import setup

setup(name='micmac',
      version='0.5',
      description='Minimally Informed CMB MAp Constructor (MICMAC) for CMB polarization data',
      author='',
      author_email='',
      url='https://github.com/CMBSciPol/MICMAC',
      packages = ['micmac'],
      package_dir = {'micmac': 'src'},
      zip_safe = False,
      entry_points = {
        'console_scripts': [
            'micmac = micmac.pipeline:__main__'
        ]      
        },
      python_requires=">=3.7"
      )
