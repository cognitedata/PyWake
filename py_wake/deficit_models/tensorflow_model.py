import numpy as np
from numpy import newaxis as na
import matplotlib.pyplot as plt

from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.examples.data import example_data_path
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
import time


class TensorFlowDeficit(WakeDeficitModel, BlockageDeficitModel):

    def __init__(self, path, set_name):
        self.surrogate = TensorflowSurrogate(path, set_name)
        WakeDeficitModel.__init__(self)

    def calc_deficit(self, WS_eff_ilk, TI_eff_ilk, dw_ijlk, dh_ijlk, hcw_ijlk, yaw_ilk, h_il, ct_ilk, **_):
        t = time.time()
        # ['yaw', 'ti', 'ct', 'x', 'y', 'z']
        inputs = [yaw_ilk[:, na], TI_eff_ilk[:, na], ct_ilk[:, na],
                  dw_ijlk, hcw_ijlk, h_il[:, na, :, na] + dh_ijlk]
        shape = np.max([v.shape for v in inputs], 0)
        inputs = np.array([np.broadcast_to(v, shape).ravel() for v in inputs]).T
        deficit = (1 - self.surrogate.predict_output(inputs).reshape(shape)) * WS_eff_ilk[:, na]
        print(time.time() - t)
        return deficit


def main():
    if __name__ == '__main__':

        wt = IEA37_WindTurbines()
        site = IEA37Site(16)
        x, y = site.initial_position.T
        deficit_model = TensorFlowDeficit(example_data_path + "rans_surrogate", "main")
        wfm = PropagateDownwind(site, wt, deficit_model, turbulenceModel=STF2017TurbulenceModel())
        sim_res = wfm(x, y, wd=270)
        sim_res.flow_map().plot_wake_map()
        plt.show()


main()
