import numpy
import py_wake
from numpy.lib.index_tricks import RClass
import inspect


class NumpyBackend():

    def __enter__(self):
        self.old_backend = py_wake.np.backend
        py_wake.np.backend = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        py_wake.np.backend = self.old_backend


class Numpy32(NumpyBackend):

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            f = getattr(numpy, name)
            if isinstance(f, (type, int, float, RClass)) or f is None or inspect.ismodule(f):
                return f
            else:
                def wrap(*args, **kwargs):
                    res = f(*args, **kwargs)
                    from numpy import dtype, float32, complex64
                    if not hasattr(res, 'dtype'):
                        return res
                    try:
                        return res.astype({dtype('float64'): float32,
                                           dtype('complex128'): complex64,
                                           }[res.dtype])
                    except KeyError:
                        # if str(res.dtype) not in ['int32', 'float32', 'int64', 'bool', '<U6']:
                        #     print(res.dtype)
                        return res
                return wrap


class Numpy64(NumpyBackend):

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            f = getattr(numpy, name)
            if isinstance(f, (type, int, float, RClass)) or f is None or inspect.ismodule(f):
                return f
            else:
                def wrap(*args, **kwargs):
                    res = f(*args, **kwargs)
                    from numpy import dtype, float32, complex128
                    if not hasattr(res, 'dtype'):
                        return res
                    try:
                        return res.astype({dtype('float64'): float,
                                           dtype('complex128'): complex128,
                                           }[res.dtype])
                    except KeyError:
                        # if str(res.dtype) not in ['int32', 'float32', 'int64', 'bool', '<U6']:
                        #     print(res.dtype)
                        return res
                return wrap


class AutogradNumpy():
    def __enter__(self):
        self.old_backend = py_wake.np.backend
        from autograd import numpy as anp
        py_wake.np.backend = anp

    def __exit__(self, exc_type, exc_val, exc_tb):
        py_wake.np.backend = self.old_backend


class NumpyWrapper():
    backend = numpy
    #backend = Numpy32()

    def __getattribute__(self, name):
        if name == 'backend':
            return object.__getattribute__(self, name)
        else:
            return getattr(self.backend, name)
