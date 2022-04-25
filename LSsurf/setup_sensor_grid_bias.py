from LSsurf import lin_op, fd_grid
import numpy as np

def setup_sensor_grid_bias(data, grids, G_data, constraint_op_list,\
                    sensor=None,\
                     spacing=None, expected_rms=None, \
                     expected_val=0, expected_rms_grad2=None ):
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
    sensor_rows=np.flatnonzero(data.sensor==sensor)
    Dsub=data[sensor_rows]
    name = f'sensor_{sensor}_bias'
    try:
        xr=[np.floor(np.min(Dsub.x)/spacing)*spacing, np.ceil(np.max(Dsub.x/spacing))*spacing]
        yr=[np.floor(np.min(Dsub.y)/spacing)*spacing, np.ceil(np.max(Dsub.y/spacing))*spacing]
        grid=fd_grid([yr, xr], spacing*np.ones(2), name=name)
    except (IndexError, ValueError):
        # grid contains no nodes
        return
    grids[name]=grid
    grid.col_0 = G_data.col_N
    grid.col_N = grid.col_0 + grid.N_nodes
    interp_mtx=lin_op(grid=grid, name=name).\
            interp_mtx(Dsub.coords()[0:2])
    interp_mtx.r=sensor_rows[interp_mtx.r.ravel()].reshape(interp_mtx.r.shape)
    G_data.add(interp_mtx)
    #Build a constraint matrix for the curvature of the bias
    if expected_rms_grad2 is not None:
        grad2_b=lin_op(grid, name='grad2_'+name).grad2(DOF=name)
        grad2_b.expected=expected_rms_grad2/np.sqrt(np.prod(grid.delta))+np.zeros(grad2_b.N_eq)
        constraint_op_list.append(grad2_b)
    #Build a constraint matrix for the magnitude of the bias
    if expected_rms is not None:
        mag_b=lin_op(grid, name='mag_'+name).data_bias(\
                ind=np.arange(grid.N_nodes),
                col=np.arange(grid.col_0, grid.col_N))
        mag_b.expected=expected_rms+np.zeros(mag_b.N_eq)
        mag_b.prior=expected_val+np.zeros(mag_b.N_eq)
        constraint_op_list.append(mag_b)
