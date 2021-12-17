from numpy import newaxis as na


def pywake2input(TI_eff_ilk, dw_ijlk, dh_ijlk, hcw_ijlk, yaw_ilk, h_il, ct_ilk, **_):
    # ['yaw', 'ti', 'ct', 'x', 'y', 'z']
    return [yaw_ilk[:, na], TI_eff_ilk[:, na], ct_ilk[:, na], dw_ijlk, hcw_ijlk, h_il[:, na, :, na] + dh_ijlk]


if __name__ == '__main__':
    import os
    from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
    from py_wake.examples.data import example_data_path
    surrogate = TensorflowSurrogate(example_data_path + "rans_surrogate", 'main')
    print(surrogate.input_channel_names)
    print(surrogate.output_channel_name)
