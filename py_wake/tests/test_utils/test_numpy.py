from py_wake import np
from py_wake.utils.numpy_utils import Numpy32
import pytest
from py_wake.tests import npt
from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.utils.layouts import rectangle, square
from py_wake.utils.profiling import timeit
from py_wake.utils.gradients import autograd


@pytest.mark.parametrize('v,dtype,dtype32', [(5., float, np.float32),
                                             (5 + 0j, complex, np.complex64)])
def test_32bit_precision(v, dtype, dtype32):

    assert np.array(v).dtype == dtype
    with Numpy32():
        assert np.array(v).dtype == dtype32
    assert np.array(v).dtype == dtype


def test_members():
    import numpy
    with Numpy32():
        assert np.pi == numpy.pi
        npt.assert_array_equal(np.r_[3, 4], numpy.r_[3, 4])


# def test_speed():
#     site = Hornsrev1Site()
#     windTurbines = V80()
#     wfm = BastankhahGaussian(site, windTurbines)
#     x, y = square(200, 5 * 80)
#     timeit(wfm.__call__, verbose=1)(x, y)
#     with Numpy32():
#         timeit(wfm.__call__, verbose=1)(x, y)

# def test_speed_gradients():
#     site = Hornsrev1Site()
#     windTurbines = V80()
#     wfm = BastankhahGaussian(site, windTurbines)
#     x, y = square(20, 5 * 80)
#     timeit(wfm.aep_gradients, verbose=1)(autograd, wrt_arg=['x', 'y'], wd_chunks=4, x=x, y=y)
#     with Numpy32():
#         timeit(wfm.aep_gradients, verbose=1)(autograd, wrt_arg=['x', 'y'], wd_chunks=4, x=x, y=y)
