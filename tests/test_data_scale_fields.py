"""
bias_functions.setup_data_field_scale_fit: the per-bias_ID coefficients on data fields
(e.g. z_x, z_y for horizontal offsets) must leave data.bias_ID one-dimensional and build
operators with one entry per datum.  pc.unique_by_rows reshapes a 1-D input in place, which
used to turn bias_ID into (N, 1) and give setup_bias_fit an N x N column array.
"""
import numpy as np
import pointCollection as pc
from LSsurf.bias_functions import assign_bias_ID, setup_bias_fit, setup_data_field_scale_fit
from LSsurf.lin_op import lin_op


def test_scale_fields_keep_bias_ID_1d():
    N = 500
    rng = np.random.default_rng(0)
    D = pc.data().from_dict({'x': rng.uniform(0, 1, N), 'y': rng.uniform(0, 1, N), 'z': rng.normal(size=N),
                             'time': np.zeros(N), 'sigma': np.ones(N), 'sigma_corr': np.ones(N),
                             'sensor': rng.integers(6, 9, N).astype(float),
                             'z_x': rng.normal(size=N), 'z_y': rng.normal(size=N),
                             'E_scale_z_x': np.full(N, 5.), 'E_scale_z_y': np.full(N, 3.)})
    D, bias_model = assign_bias_ID(D, ['sensor'])
    G = lin_op(name='test', col_0=0, col_N=10)
    constraints = []
    setup_bias_fit(D, bias_model, G, constraints, bias_param_name='bias_ID')
    setup_data_field_scale_fit(D, bias_model, G, constraints, {'data_scale_fields': ['z_x', 'z_y']})
    assert D.bias_ID.shape == (N,)
    # one bias and two scale coefficients per datum
    assert G.r.shape == (3*N,) and G.c.shape == (3*N,)
    n_id = len(np.unique(D.bias_ID))
    assert [op.expected.tolist() for op in constraints[1:]] == [[5.]*n_id, [3.]*n_id]
    # the scale columns carry the field values
    cols_x = bias_model['bias_ID_dict']
    for key, entry in cols_x.items():
        rows = np.flatnonzero(G.c == entry['col_z_x'])
        assert np.allclose(np.sort(G.v[rows]), np.sort(D.z_x[D.bias_ID == key]))
