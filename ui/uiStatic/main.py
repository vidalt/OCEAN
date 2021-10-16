# Author: Moises Henrique Pereira
# this class is used to run the application

import sys
import os

# these two lines is needed because the files to generate the counterfactual are inside the src
path0 = sys.path[0]
sys.path.insert(0,os.path.join(path0, '..', '..', 'src'))
sys.path.insert(1,os.path.join(path0, '..'))

from MainApplication.MainApplication import MainApplication

def main():
    application = MainApplication()
    application.run()

if __name__ == '__main__':
    main()