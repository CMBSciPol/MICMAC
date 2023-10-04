from distutils.core import setup

setup(name='non_parametric_ML_compsep',
      version='0.3',
      description='Non-parametric maximum likelihood component separation for CMB polarization data',
      author='',
      author_email='',
      url='https://github.com/CMBSciPol/Pixel-Non-Parametric-CompSep',
      packages = ['non_parametric_ML_compsep'],
      package_dir = {'non_parametric_ML_compsep': 'src'},
      zip_safe = False,
      entry_points = {
        'console_scripts': [
            'non_parametric_ML_compsep = ncp.pipeline:__main__'
        ]      
        },
      python_requires=">=3.7"
      )
