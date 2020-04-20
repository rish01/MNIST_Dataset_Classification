from time import clock
from functools import wraps

def time_me(f):
    """Times a function and prints out how long it took to run.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        init_time = clock()
        output = f(*args, **kwargs)
        print(('\'{}\' execution time: {:3.4f}s'.format(f.__name__, clock()-init_time)))
        return output
    return wrapper