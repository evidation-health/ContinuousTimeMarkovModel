import time, inspect
try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner


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

class timewith():
    def __init__(self, name=''):
        self.name = name
        self.start = time.time()
        self.prev = time.time()

    @property
    def elapsed(self):
        return time.time() - self.start

    @property
    def delta(self):
        cur = time.time()
        diff = cur - self.prev
        self.prev = cur
        return diff

    def checkpoint(self, name=''):
        print '{timer} {checkpoint} at {elapsed} took {delta} seconds'.format(
            timer=self.name,
            checkpoint=name,
            elapsed=self.elapsed,
            delta=self.delta,
        ).strip()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint('finished')
        pass
