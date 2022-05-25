import numpy
import py_wake
from numpy.lib.index_tricks import RClass
import inspect
import autograd.numpy as autograd_numpy


class NumpyBackend():

    def __enter__(self):
        self.old_backend = py_wake.np.backend
        py_wake.np.backend = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        py_wake.np.backend = self.old_backend


class Numpy32(NumpyBackend):
    backend = numpy

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            f = getattr(self.backend, name)
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


class AutogradNumpy32(Numpy32):
    backend = autograd_numpy


class AutogradNumpy():
    def __enter__(self):
        self.old_backend = py_wake.np.backend
        if isinstance(self.old_backend, Numpy32):
            py_wake.np.backend = AutogradNumpy32()
        else:
            py_wake.np.backend = autograd_numpy

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
