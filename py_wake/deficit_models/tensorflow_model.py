import numpy as np
from numpy import newaxis as na
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
from py_wake.examples.data import rans_surrogate
from py_wake.examples.data import example_data_path
import inspect


def normalized_deficit2deficit(output, WS_eff_ilk, **_):
    return output * WS_eff_ilk[:, na]


class TensorFlowDeficit(WakeDeficitModel, BlockageDeficitModel):

    def __init__(self, path=example_data_path + "rans_surrogate", set_name="main",
                 get_input=rans_surrogate.pywake2input,
                 output2deficit=normalized_deficit2deficit):
        self.surrogate = TensorflowSurrogate(path, set_name)
        self.args4deficit = (set(inspect.getfullargspec(get_input).args) |
                             set(inspect.getfullargspec(output2deficit).args) - {'output', 'self'})
        WakeDeficitModel.__init__(self)
        BlockageDeficitModel.__init__(self)
        self.get_input = get_input
        self.output2deficit = output2deficit

    def calc_deficit(self, **kwargs):
        inputs = self.get_input(**kwargs)
        shape = np.max([v.shape for v in inputs], 0)
        inputs = np.array([np.broadcast_to(v, shape).ravel() for v in inputs]).T
        deficit = self.output2deficit(self.surrogate.predict_output(inputs).reshape(shape), **kwargs)
        return deficit


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.wind_farm_models.engineering_models import All2AllIterative
        from py_wake.turbulence_models.stf import STF2017TurbulenceModel
        from py_wake.flow_map import XYGrid
        from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
        wt = DTU10MW()
        D = wt.diameter()
        site = IEA37Site(16)
        x, y = site.initial_position[:1].T
        deficit_model = TensorFlowDeficit(example_data_path + "rans_surrogate", "main")
        wfm = All2AllIterative(site, wt, deficit_model, blockage_deficitModel=deficit_model,
                               turbulenceModel=STF2017TurbulenceModel())
        sim_res = wfm(x, y, wd=270, ws=10)
        x, y = np.linspace(-500, 5000, 500), np.linspace(-500, 500, 500)
        fm = sim_res.flow_map(XYGrid(x, y))
        fm.plot_wake_map(normalize_with=D)

        plt.figure()
        for d in [1, 3, 5]:
            fm.WS_eff.interp(x=d * D).plot(label=d)
        plt.legend()
        plt.grid()
        plt.show()


main()
