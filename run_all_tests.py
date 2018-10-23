# This is a script to run all the tests. Can be helpful to run before pushing your code.

import subprocess
import os
import time


def run_all_tests():
    files_to_test = [os.path.join(os.path.abspath(os.getcwd()), 'tests', f) for f in os.listdir('./tests')
                     if f.startswith('test')]
    
    for file in files_to_test:
        run_test(file)


def run_test(filename):
    """
    Run a particular test defined by the filename.
    """
    print('Running test %s' % os.path.basename(filename))
    t_start = time.time()
    subprocess.call(["python", filename])
    t_end = time.time()
    print('Execution time for test %s is %d seconds.' % (os.path.basename(filename), t_end - t_start))


if __name__ == '__main__':
    run_all_tests()
