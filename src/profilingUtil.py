import time, inspect

def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        theArgs = inspect.getcallargs(f,*args)
        if 'self' in theArgs:
            print theArgs['self'].__class__.__name__, f.__name__, 'took', end - start, 'sec' 
        else:
            print f.__name__, 'took', end - start, 'sec'
        return result
    return f_timer
