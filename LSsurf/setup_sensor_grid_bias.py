from LSsurf import lin_op, fd_grid
import numpy as np

def setup_sensor_grid_bias(sensor, data, G_data, constraint_op_list, grids, \
                     spacing=None, expected_rms=None, \
                     expected_value=0, expected_rms_grad=None ):
    '''
    set up a matrix to fit a gridded bias that multiplies a data parameter
    
    Parameters:
        data: pointCollection.data object containing the data
        G_data: fd_grid.operator that maps model parameters to the data
        constraint_op_list: list: constraint operators
        grids: dict: named grids representing model parameters
        expected_rms: float, default=None: Expected RMS value of the parameter. If not specified, the RMS of the parameter will be unconstrained
        expected_value: float, default=0: Expected value of the parameter.
        expected_rms_grad: float, default=None: expected RMS gradient of the parameter.  If not specified, the gradient of the parameter will be unconstrained
    '''
    Dsub=data[data.sensor==sensor]
    name = f'sensor_{sensor}_bias'
    xr=[np.floor(np.min(Dsub.x)/spacing)*spacing, np.ceil(np.max(Dsub.x/spacing))*spacing]
    yr=[np.floor(np.min(Dsub.x)/spacing)*spacing, np.ceil(np.max(Dsub.y/spacing))*spacing]
    grid=fd_grid([yr, xr], spacing*np.ones(2), name=name)
    grids[name]=grid
    grid.col_0 = G_data.col_N
    grid.col_N = grid.col_0 + grid.N_nodes
    interp_mtx=lin_op(grid=grid, name=name).\
            interp_mtx(Dsub.coords()[0:2])
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
