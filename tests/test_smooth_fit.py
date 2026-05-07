import numpy as np
from LSsurf.smooth_fit import smooth_fit
import pointCollection as pc
import os


def run_fit():

    # define the domain's width in x, y, and time
    W_d = {'x': 1.e4, 'y': 400, 't': 2}
    # define the grid center:
    ctr = {'x': 0., 'y': 0., 't': 0.}
    # define the grid spacing
    spacing = {'z0': 100, 'dz': 200, 'dt': 0.25}

    # define the data as a sine wave with a wavelength of 2 km and an amplitude of 100m.
    x = np.arange(-W_d['x']/2, W_d['x']/2, 100)
    lambda_x = 2000
    amp = 100

    # define data for two time steps
    D0 = pc.data().from_dict({'x': x, 'y': np.zeros_like(x), 'z': np.zeros_like(x),
                              'time': np.zeros_like(x)-0.99, 'sigma': np.zeros_like(x)+1})
    D1 = pc.data().from_dict({'x': x, 'y': np.zeros_like(x), 'z': amp-amp*np.cos(2*np.pi*x/lambda_x),
                              'time': np.zeros_like(x)+0.99, 'sigma': np.zeros_like(x)+1})
    data_dt = pc.data().from_list([D0, D1])

    # define constraint magnituds
    data_gap_scale = 2500
    E_RMS = {}  
    E_RMS['d3z_dx2dt'] = 0.006
    E_RMS['d2z_dxdt'] = 0.006*data_gap_scale
    E_RMS['d2z0_dx2'] = 0.03
    E_RMS['dz0_dx'] = 0.03*data_gap_scale
    E_RMS['d2z_dt2'] = 6000

    data_dt_gap = data_dt[np.abs(data_dt.x) > 1000]
    W_fit = {'x': 10000.0, 'y': 400, 't': 2}
    S = smooth_fit(data = data_dt_gap, 
                   ctr = ctr, 
                   W = W_fit, 
                   spacing = spacing, 
                   E_RMS = E_RMS,
                   reference_epoch=4,  
                   compute_E=True,
                   max_iterations=1, 
                   dzdt_lags=[1])
    return S


def save_fit():

    # this function will replace the output of the test function if it does not exist.
    # After this happens the tests will succeed in an unenlightening way.

    # check if the output files exist.
    file_count = 0
    for group in ['z0', 'dz']:
        dst_file = os.path.join('test_data', group+'.h5')
        if os.path.isfile(dst_file):
            file_count += 1

    # if all files exist, return
    if file_count == 2:
        return

    print("regenerating the fit:")
    S = run_fit()

    if not os.path.isdir('test_data'):
        os.mkdir('test_data')

    for group in ['z0', 'dz']:
        dst_file = os.path.join('test_data', group+'.h5')
        S['m'][group].to_h5(dst_file)

    return


def test_fit():

    # refresh the saved files if needed:
    save_fit()

    # run the fit:
    S = run_fit()

    # load the saved data
    m_ref = {name: pc.data().from_h5(os.path.join('test_data', name+'.h5'))
             for name in ['z0', 'dz']}

    # make sure the saved data matches the fit model
    for group, ref_data in m_ref.items():
        test_data = S['m'][group]
        for field in m_ref[group].fields:
            assert np.allclose(getattr(test_data, field),
                               getattr(ref_data, field),
                               equal_nan=True, atol=0.01, rtol=0.1), \
                f"excessive misfit for field {field} in group {group}"
