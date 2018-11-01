import sys
import time


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.ctime()
        if kwargs:
            print('Starting {0} for N={1}, endtime={2} at {3}'.format(
                sys.argv[0],
                kwargs['N'],
                kwargs['t_end'],
                start_time))
        else:
            print('Starting {0} at {1}'.format(
                sys.argv[0],
                start_time))

        start_time = time.time()

        func(*args, **kwargs)

        print('Finished at {0}'.format(time.ctime()))

        elapsed = int(time.time() - start_time)
        print('ELAPSED TIME = {:02d}:{:02d}:{:02d}'.format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        print("******************************************************************************")
    return wrapper
