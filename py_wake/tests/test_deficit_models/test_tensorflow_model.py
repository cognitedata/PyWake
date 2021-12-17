import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.flow_map import XYGrid
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake.examples.data import example_data_path
from py_wake.deficit_models.tensorflow_model import TensorFlowDeficit
from py_wake.tests import npt
import os
from py_wake.tests.test_files import tfp


def test_DTU10MW_surrogate():
    wt = DTU10MW()
    D = wt.diameter()
    site = IEA37Site(16, ti=0.1)
    x, y = site.initial_position[:1].T
    deficit_model = TensorFlowDeficit(example_data_path + "rans_surrogate", "main")
    wfm = All2AllIterative(site, wt, deficit_model, blockage_deficitModel=deficit_model,
                           turbulenceModel=STF2017TurbulenceModel())

    sim_res = wfm(x, y, wd=270, ws=10)
    x, y = np.linspace(-400, 4000, 400), np.linspace(-400, 400, 500)

    z_lst = [10, wt.hub_height(), 200]
    d_lst = np.array([-1, 1, 3, 5, 10])

    ref_path = tfp + "DTU10MW_RANS_LUT_ref.nc"
    if not os.path.isfile(ref_path):
        ds = xr.open_dataset(r'c:\tmp\db_yaw2deg_8cD.nc').U.sel(yaw=0, ti=sim_res.TI.item(), ct=sim_res.CT.item())
        shear = ds.isel(y=0, x=0)
        #ds = ds - shear
        ds.interp(z=z_lst, y=y, x=d_lst * D).to_netcdf(ref_path)
    ds_ref = xr.load_dataarray(ref_path)

    for z in z_lst:
        plt.figure()
        plt.title(f"Height = {z}")

        plt.plot([], [], 'k', label='Surrogate')
        plt.plot([], [], 'k--', label='LUT')
        for d in d_lst:
            ws = sim_res.flow_map(XYGrid(x=d * D, y=y, h=z)).WS_eff
            c = plt.plot(ws.y / D, ws.squeeze(), label=f'{d}D downstream')[0].get_color()
            ws_ref = ((1 - ds_ref.sel(z=z, x=d * D)) * 10)
            plt.plot(ws_ref.y / D, ws_ref.squeeze(), '--', color=c)
            npt.assert_allclose(ws.squeeze(), ws_ref.squeeze())
        plt.legend(loc="lower right")
        plt.grid()

    if 0:
        plt.show()
