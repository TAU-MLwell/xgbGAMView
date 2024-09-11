from distutils.core import setup
setup(
  name = 'xgbGAMView',     
  packages = ['xgbGAMView'],   
  version = '0.1',      
  license='MIT',        
  description = 'Generalized Additive Models (GAMs) trained using xgboost with visualization and smoothing',   # Give a short description about your library
  author = 'MLWell lab, Tel-Aviv University',                   # Type in your name
  author_email = 'rgb@tauex.tau.ac.il',      # Type in your E-Mail
  url = 'https://github.com/TAU-MLwell/xgbGAMView',  
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['Generalized Additive Model', 'GAM', 'explainable AI', "xAI"],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package

    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',

    'License :: OSI Approved :: MIT License',   

    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)