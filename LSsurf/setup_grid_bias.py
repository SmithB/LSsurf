from LSsurf import lin_op
import numpy as np

def setup_grid_bias(data, G_data, constraint_op_list, grids, \
                     name=None, param=None, grid=None, expected_rms=None, \
                     expected_value=None, expected_rms_grad=None ):
    '''
    set up a matrix to fit a gridded bias that multiplies a data parameter
    
    Parameters:
        data: pointCollection.data object containing the data
        G_data: fd_grid.operator that maps model parameters to the data
        constraint_op_list: list: constraint operators
        grids: dict: named grids representing model parameters
        expected: float, default=None: Expected RMS value of the parameter. If not specified, the RMS of the parameter will be unconstrained
        expected_value: float, default=0: Expected value of the parameter.
        expected_rms_grad: float, default=None: expected RMS gradient of the parameter.  If not specified, the RMS of the parameter will be unconstrained
    '''
    if name is None:
        name=param+'_bias'
    grids[name]=grid
    grid.col_0 = G_data.col_N
    grid.col_N = grid.col_0 + grid.N_nodes
    interp_mtx=lin_op(grid=grid, name=name).\
            interp_mtx(data.coords()[0:2])
    # operator values are equal to the interpolator matrix values times the parameter values
    temp = interp_mtx.v.ravel() * getattr(data, param)[interp_mtx.r.ravel().astype(int)]
    interp_mtx.v = temp.reshape(interp_mtx.v.shape)
    G_data.add(interp_mtx)
    #Build a constraint matrix for the curvature of the bias
    if expected_rms_grad is not None:
        grad2_param=lin_op(grid, name='grad2_'+name).grad2(DOF=param)
        grad2_param.expected=expected_rms_grad+np.zeros(grad2_param.N_eq)/\
            np.sqrt(np.prod(grid.delta))
        constraint_op_list.append(grad2_param)
    #Build a constraint matrix for the magnitude of the bias
    if expected_rms is not None:
        mag_param=lin_op(grid, name='mag_'+name).data_bias(\
                ind=np.arange(grid.N_nodes),
                col=np.arange(grid.col_0, grid.col_N))
        mag_param.expected=expected_rms+np.zeros(mag_param.N_eq)
        mag_param.expected_val=expected_value+np.zeros(mag_param.N_eq)
        constraint_op_list.append(mag_param)
