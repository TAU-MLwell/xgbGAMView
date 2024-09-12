from distutils.core import setup
setup(
  name = 'xgbGAMView',     
  packages = ['xgbGAMView'],   
  version = '0.1',      
  license='MIT',        
  description = 'Generalized Additive Models (GAMs) trained using xgboost with visualization and smoothing',   
  author = 'MLWell lab, Tel-Aviv University', 
  author_email= 'rgb@tauex.tau.ac.il',
  url = 'https://github.com/TAU-MLwell/xgbGAMView',  
  download_url = 'https://github.com/TAU-MLwell/xgbGAMView/archive/refs/tags/0.1.tar.gz',   
  keywords = ['Generalized Additive Model', 'GAM', 'explainable AI', "xAI"],  
  install_requires=[            
        'numpy',
        'pandas',
        'matplotlib',
        'xgboost',
        'sklearn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',    

    'Intended Audience :: Developers',    
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License'],
  python_requires='>=3.8'
)
