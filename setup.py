from distutils.core import setup

setup(name='micmac',
      version='0.3',
      description='Non-parametric maximum likelihood component separation for CMB polarization data',
      author='',
      author_email='',
      url='https://github.com/CMBSciPol/Pixel-Non-Parametric-CompSep',
      packages = ['micmac'],
      package_dir = {'micmac': 'src'},
      zip_safe = False,
      entry_points = {
        'console_scripts': [
            'micmac = ncp.pipeline:__main__'
        ]      
        },
      python_requires=">=3.7"
      )
