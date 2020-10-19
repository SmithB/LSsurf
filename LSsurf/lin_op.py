# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:25:46 2017

@author: ben
"""
import numpy as np
import scipy.sparse as sp
from LSsurf.fd_grid import fd_grid


class lin_op:
    def __init__(self, grid=None, row_0=0, col_N=None, col_0=None, name=None):
        # a lin_op is an operator that represents a set of linear equations applied
        # to the nodes of a grid (defined in fd_grid.py)
        if col_0 is not None:
            self.col_0=col_0
        elif grid is not None:
            self.col_0=grid.col_0
        self.col_N=None
        if col_N is not None:
            self.col_N=col_N
        elif grid is not None:
            self.col_N=grid.col_N
        self.row_0=row_0
        self.N_eq=0
        self.name=name
        self.id=None
        self.r=np.array([], dtype=int)
        self.c=np.array([], dtype=int)
        self.v=np.array([], dtype=float)
        self.ind0=np.zeros([0], dtype=int)
        self.TOC={'rows':dict(),'cols':dict()}
        self.grid=grid
        self.dst_grid=None
        self.dst_ind0=None
        self.expected=None
        self.shape=None
        self.size=None

    def __update_size_and_shape__(self):
        self.shape = (self.N_eq, self.col_N)


    def diff_op(self, delta_subs, vals,  which_nodes=None, valid_equations_only=True):
        # build an operator that calculates linear combination of the surrounding
        # values at each node of a grid.
        # A template, given by delta_subs and vals contains a list of offsets
        # in each direction of the grid, and a list of values corresponding
        # to each offset.  Only those nodes for which the template falls
        # entirely inside the grid are included in the operator

        if valid_equations_only:
            # compute the maximum and minimum offset in each dimension.  These
            # will be used to eliminate equations that extend outside the model
            # domain
            max_deltas=[np.max(delta_sub) for delta_sub in delta_subs]
            min_deltas=[np.min(delta_sub) for delta_sub in delta_subs]
        else:
            # treat the maximum and minimum offset in each dimension as zero,
            # so no equations are truncated
            max_deltas=[0 for delta_sub in delta_subs]
            min_deltas=[0 for delta_sub in delta_subs]
        #generate the center-node indices for each calculation
        # if in dimension k, min_delta=-a and max_delta = +b, the number of indices is N,
        # then the first valid center is a and the last is N-b
        sub0s=np.meshgrid(*[np.arange(np.maximum(0, -min_delta), np.minimum(Ni, Ni-max_delta)) for Ni, min_delta, max_delta in zip(self.grid.shape, min_deltas, max_deltas)], indexing='ij')
        sub0s=[sub.ravel() for sub in sub0s]
        if which_nodes is not None:
            temp_mask=np.in1d(self.grid.global_ind(sub0s), which_nodes)
            sub0s=[temp[temp_mask] for temp in sub0s]
        self.r, self.c=[np.zeros((len(sub0s[0]), len(delta_subs[0])), dtype=int) for _ in range(2)]
        self.v=np.zeros_like(self.r, dtype=float)
        self.N_eq=len(sub0s[0])
        # loop over offsets
        for ii in range(len(delta_subs[0])):
            # build a list of subscripts over dimensions
            this_sub=[sub0+delta[ii] for sub0, delta in zip(sub0s, delta_subs)]
            self.r[:,ii]=self.row_0+np.arange(0, self.N_eq, dtype=int)
            if valid_equations_only:
                self.c[:,ii]=self.grid.global_ind(this_sub)
                self.v[:,ii]=vals[ii].ravel()
            else:
                # need to remove out-of-bound subscripts
                self.c[:,ii], valid_ind=self.grid.global_ind(this_sub, return_valid=True)
                self.v[:,ii]=vals[ii].ravel()*valid_ind.ravel()
        #if not valid_equations_only: [Leave this commented until it causes a problem]
        #    # remove the elements that have v=0
        #    nonzero_v = self.v.ravel() != 0
        #    self.r = self.r.ravel()[nonzero_v]
        #    self.c = self.c.ravel()[nonzero_v]
        #    self.v = self.v.ravel()[nonzero_v]
        self.ind0 = self.grid.global_ind(sub0s).ravel()
        self.TOC['rows'] = {self.name:range(self.N_eq)}
        self.TOC['cols'] = {self.grid.name:np.arange(self.grid.col_0, self.grid.col_0+self.grid.N_nodes)}
        self.__update_size_and_shape__()
        return self

    def add(self, op):
        # combine a set of operators into a composite operator by adding them.
        # the same thing could be accomplished by converting the operators to
        # sparse arrays and adding the arrays, but this method keeps track of the
        # table of contents for the operators.
        # if a list of operators is provided, all are added together, or a single
        # operator can be added to an existing operator.
        if isinstance(op, list) or isinstance(op, tuple):
            for this_op in op:
                op.add(self, this_op)
            return self
        if self.r is not None:
            self.r=np.append(self.r, op.r)
            self.c=np.append(self.c, op.c)
            self.v=np.append(self.v, op.v)
            self.ind0=np.append(self.ind0, op.ind0)
        else:
            self.r=op.r
            self.c=op.c
            self.v=op.v
            self.ind0=op.ind0
        # assume that the new op may have columns that aren't in self.cols, and
        # add any new columns to the table of contents
        for key in op.TOC['cols'].keys():
            self.TOC['cols'][key]=op.TOC['cols'][key]
        self.col_N=np.maximum(self.col_N, op.col_N)
        self.__update_size_and_shape__()
        return self

    def interp_mtx(self, pts):
        # create a matrix that, when it multiplies a set of nodal values,
        # gives the bilinear interpolation between those nodes at a set of
        # data points
        pts=[pp.ravel() for pp in pts]
        # Identify the nodes surrounding each data point
        # The floating-point subscript expresses the point locations in terms
        # of their grid positions
        ii=self.grid.float_sub(pts)
        cell_sub=self.grid.cell_sub_for_pts(pts)
        # calculate the fractional part of each cell_sub
        i_local=[a-b for a, b in zip(ii,cell_sub)]
        # find the index of the node below each data point
        global_ind=self.grid.global_ind(cell_sub)
        # make a list of dimensions based on the dimensions of the grid
        if self.grid.N_dims==1:
            list_of_dims=np.mgrid[0:2]
        elif self.grid.N_dims==2:
            list_of_dims=np.mgrid[0:2, 0:2]
        elif self.grid.N_dims==3:
            list_of_dims=np.mgrid[0:2, 0:2, 0:2]
        delta_ind=np.c_[[kk.ravel() for kk in list_of_dims]]
        n_neighbors=delta_ind.shape[1]
        Npts=len(pts[0])
        rr=np.zeros([Npts, n_neighbors])
        cc=np.zeros([Npts, n_neighbors])
        vv= np.ones([Npts, n_neighbors])
        # make lists of row and column indices and weights for the nodes
        for ii in range(n_neighbors):
            rr[:,ii]=np.arange(len(pts[0]))
            cc[:,ii]=global_ind+np.sum(self.grid.stride*delta_ind[:,ii])
            for dd in range(self.grid.N_dims):
                if delta_ind[dd, ii]==0:
                    vv[:,ii]*=(1.-i_local[dd])
                else:
                    vv[:,ii]*=i_local[dd]
        self.r=rr
        self.c=cc
        self.v=vv
        self.N_eq=Npts
        # in this case, sub0s is the index of the data points
        self.ind0=np.arange(0, Npts, dtype='int')
        # report the table of contents
        self.TOC['rows']={self.name:np.arange(self.N_eq, dtype='int')}
        self.TOC['cols']={self.grid.name:np.arange(self.grid.col_0, self.grid.col_0+self.grid.N_nodes)}
        self.__update_size_and_shape__()
        return self

    def grad(self, DOF='z'):
        coeffs=np.array([-1., 1.])/(self.grid.delta[0])
        dzdx=lin_op(self.grid, name='d'+DOF+'_dx').diff_op(([0, 0],[-1, 0]), coeffs)
        dzdy=lin_op(self.grid, name='d'+DOF+'_dy').diff_op(([-1, 0],[0, 0]), coeffs)
        self.vstack((dzdx, dzdy))
        self.__update_size_and_shape__()
        return self

    def grad_dzdt(self, DOF='z', t_lag=1):
        coeffs=np.array([-1., 1., 1., -1.])/(t_lag*self.grid.delta[0]*self.grid.delta[2])
        d2zdxdt=lin_op(self.grid, name='d2'+DOF+'_dxdt').diff_op(([ 0, 0,  0, 0], [-1, 0, -1, 0], [-t_lag, -t_lag, 0, 0]), coeffs)
        d2zdydt=lin_op(self.grid, name='d2'+DOF+'_dydt').diff_op(([-1, 0, -1, 0], [ 0, 0,  0, 0], [-t_lag, -t_lag, 0, 0]), coeffs)
        self.vstack((d2zdxdt, d2zdydt))
        self.__update_size_and_shape__()
        return self

    def diff(self, lag=1, dim=0):
        coeffs=np.array([-1., 1.])/(lag*self.grid.delta[dim])
        deltas=[[0, 0] for this_dim in range(self.grid.N_dims)]
        deltas[dim]=[0, lag]
        self.diff_op((deltas), coeffs)
        self.__update_size_and_shape__()
        return self

    def dzdt(self, lag=1, DOF='dz'):
        coeffs=np.array([-1., 1.])/(lag*self.grid.delta[2])
        self.diff_op(([0, 0], [0, 0], [0, lag]), coeffs)
        self.__update_size_and_shape__()
        self.update_dst_grid([0, 0, 0.5*lag*self.grid.delta[2]], np.array([1, 1, 1]))
        return self

    def d2z_dt2(self, DOF='dz', t_lag=1):
        coeffs=np.array([-1, 2, -1])/((t_lag*self.grid.delta[2])**2)
        self=lin_op(self.grid, name='d2'+DOF+'_dt2').diff_op(([0,0,0], [0,0,0], [-t_lag, 0, t_lag]), coeffs)
        self.__update_size_and_shape__()
        return self

    def grad2(self, DOF='z'):
        coeffs=np.array([-1., 2., -1.])/(self.grid.delta[0]**2)
        d2zdx2=lin_op(self.grid, name='d2'+DOF+'_dx2').diff_op(([0, 0, 0],[-1, 0, 1]), coeffs)
        d2zdy2=lin_op(self.grid, name='d2'+DOF+'_dy2').diff_op(([-1, 0, 1],[0, 0, 0]), coeffs)
        d2zdxdy=lin_op(self.grid, name='d2'+DOF+'_dxdy').diff_op(([-1, -1, 1,1],[-1, 1, -1, 1]), 0.5*np.array([-1., 1., 1., -1])/(self.grid.delta[0]**2))
        self.vstack((d2zdx2, d2zdy2, d2zdxdy))
        self.__update_size_and_shape__()
        return self

    def grad2_dzdt(self, DOF='z', t_lag=1):
        coeffs=np.array([-1., 2., -1., 1., -2., 1.])/(t_lag*self.grid.delta[0]**2.*self.grid.delta[2])
        d3zdx2dt=lin_op(self.grid, name='d3'+DOF+'_dx2dt').diff_op(([0, 0, 0, 0, 0, 0],[-1, 0, 1, -1, 0, 1], [-t_lag,-t_lag,-t_lag, 0, 0, 0]), coeffs)
        d3zdy2dt=lin_op(self.grid, name='d3'+DOF+'_dy2dt').diff_op(([-1, 0, 1, -1, 0, 1], [0, 0, 0, 0, 0, 0], [-t_lag, -t_lag, -t_lag, 0, 0, 0]), coeffs)
        coeffs=np.array([-1., 1., 1., -1., 1., -1., -1., 1.])/(self.grid.delta[0]**2*self.grid.delta[2])
        d3zdxdydt=lin_op(self.grid, name='d3'+DOF+'_dxdydt').diff_op(([-1, 0, -1, 0, -1, 0, -1, 0], [-1, -1, 0, 0, -1, -1, 0, 0], [-t_lag, -t_lag, -t_lag, -t_lag, 0, 0, 0, 0]),  coeffs)
        self.vstack((d3zdx2dt, d3zdy2dt, d3zdxdydt))
        self.__update_size_and_shape__()
        return self

    def normalize_by_unit_product(self, wt=1):
        # normalize an operator by its magnitude's product with a vector of ones.
        # optionally rescale the result by a factor of wt
        unit_op=lin_op(col_N=self.col_N)
        unit_op.N_eq=self.N_eq
        unit_op.r, unit_op.c, unit_op.v = [self.r, self.c, np.abs(self.v)]
        unit_op.__update_size_and_shape__()
        norm = unit_op.toCSR(row_N=unit_op.N_eq).dot(np.ones(self.shape[1]))
        scale = np.zeros_like(norm)
        scale[norm>0] = 1./norm[norm>0]
        self.v *= scale[self.r]*wt

    def mean_of_bounds(self, bds, mask=None):
        # make a linear operator that calculates the mean of all points
        # in its grid that fall within bounds specified by 'bnds',  If an
        # empty matrix is specified for a dimension, the entire dimension is
        # included.
        # optionally, a 'mask' variable can be used to select from within the
        # bounds.

        coords=np.meshgrid(*self.grid.ctrs, indexing='ij')
        in_bds=np.ones_like(coords[0], dtype=bool)
        for dim, bnd in enumerate(bds):
            if bds[dim] is not None:
                in_bds=np.logical_and(in_bds, np.logical_and(coords[dim]>=bnd[0], coords[dim] <= bnd[1]));
        if mask is not None:
            in_bds=np.logical_and(in_bds, mask)
        self.c=self.grid.global_ind(np.where(in_bds))
        self.r=np.zeros(in_bds.ravel().sum(), dtype=int)
        self.v=np.ones(in_bds.ravel().sum(), dtype=float)/np.sum(in_bds.ravel())
        self.TOC['rows']={self.name:self.r}
        self.TOC['cols']={self.name:self.c}
        self.N_eq=1.
        self.__update_size_and_shape__()
        return self

    def mean_of_mask(self, mask, dzdt_lag=None):
        # make a linear operator that takes the mean of points multiplied by
        # a 2-D mask.  If the grid has a time dimension, the operator takes the
        # mean for each time slice.  If dzdt_lags are provided, it takes the
        # mean dzdt as a function of time
        coords=np.meshgrid(*self.grid.ctrs[0:2], indexing='ij')
        mask_g=mask.interp(coords[1], coords[0])
        mask_g[~np.isfinite(mask_g)]=0
        i0, j0 = np.nonzero(mask_g)
        if self.grid.cell_area is None:
            v0=mask_g.ravel()[np.flatnonzero(mask_g)]
        else:
            v0=(mask_g*self.grid.cell_area).ravel()[np.flatnonzero(mask_g)]
        v0 /= v0.sum()
        y0=np.sum((self.grid.bds[0][0]+i0.ravel()*self.grid.delta[0])*v0)
        x0=np.sum((self.grid.bds[1][0]+j0.ravel()*self.grid.delta[1])*v0)
        
        if len(self.grid.shape) < 3:
            # no time dimension: Just average the grid
            self.r = np.zeros_like(i0)
            self.c = self.grid.global_ind([i0,j0])
            self.v = v0
            self.N_eq=1
            self.col_N=np.max(self.c)+1
            self.__update_size_and_shape__()
            self.dst_grid = fd_grid( [[y0, y0], [x0, x0]], \
                                self.grid.delta, 0,  col_N=0,
                                srs_proj4=self.grdi.srs_proj4)
            self.dst_ind0 = np.array([0]).astype(int)
            return self
        rr, cc, vv = [[],[], []]
        if dzdt_lag is None:
            # average each time slice
            for ii in range(self.grid.shape[2]):
                rr += [np.zeros_like(i0)+ii]
                cc += [self.grid.global_ind([i0, j0, np.zeros_like(i0)+ii])]
                vv += [v0]
            t_vals=self.grid.ctrs[2]
        else:
            for ii in range(self.grid.shape[2]-dzdt_lag):
                for dLag in [0, dzdt_lag]:
                    rr += [np.zeros_like(i0)+ii]    
                    cc += [self.grid.global_ind([i0, j0, np.zeros_like(i0) + dLag])]
                    if dLag==0:
                        vv += [-v0/dzdt_lag/self.grid.delta[2]]
                    else:
                        vv += [v0/dLag/self.grid.delta[2]]
            t_vals=self.grid.ctrs[-1][:-dzdt_lag] + self.grid.delta[-1]*dzdt_lag/2
        self.r, self.c, self.v = [ np.concatenate(ii) for ii in [rr, cc, vv]] 
        self.dst_grid = fd_grid( [[y0, y0], [x0, x0], [t_vals[0], t_vals[-1]]], \
                                self.grid.delta, 0,  col_N=self.r.max()+1,
                                srs_proj4=self.grid.srs_proj4)
        self.N_eq = self.r.max()+1
        self.__update_size_and_shape__()
        self.dst_ind0=np.arange(self.N_eq, dtype=int)
        return self

    def sum_to_grid3(self, kernel_size,  sub0s=None, lag=None, taper=True, valid_equations_only=True, dims=None):
        # make an operator that adds values with a kernel of size kernel_size pixels
        # centered on the grid cells identified in sub0s
        # optional: specify 'lag' to compute a dz/dt
        # specify taper=True to include half-weights on edge points and
        #          quarter-weights on corner points
        if taper:
            half_kernel=np.floor((kernel_size-1)/2).astype(int)
        else:
            half_kernel=np.floor(kernel_size/2).astype(int)

        if dims is None:
            dims=range(len(self.grid.shape))
            n_dims=len(dims)
        else:
            n_dims=len(dims)

        if sub0s is None:
            if taper:
                if valid_equations_only:
                    sub0s = np.meshgrid( *[np.arange(half_kernel+1, self.grid.shape[ii], kernel_size[ii]-1, dtype=int) for ii in dims], indexing='ij')
                else:
                    sub0s = np.meshgrid( *[np.arange(0, self.grid.shape[ii]+1, kernel_size[ii]-1, dtype=int) for ii in dims], indexing='ij')
            else:
                sub0s = np.meshgrid(*[np.arange(half_kernel, self.grid.shape[ii], kernel_size[ii], dtype=int) for ii in dims], indexing='ij')
        ind0 = self.grid.global_ind(sub0s[0:n_dims])

        if np.mod(kernel_size[0]/2,1)==0:
            # even case
            di, dj = np.meshgrid(np.arange(-half_kernel[0], half_kernel[0]),\
                             np.arange(-half_kernel[1], half_kernel[1]), indexing='ij')
            grid_shift=[-self.grid.delta[0]/2, -self.grid.delta[1]/2, 0][0:n_dims]
        else:
            # odd_case
            di, dj = np.meshgrid(np.arange(-half_kernel[0], half_kernel[0]+1),\
                             np.arange(-half_kernel[1], half_kernel[1]+1), indexing='ij')
            grid_shift=[0, 0, 0][0:len(dims)]

        # make the weighting matrix:
        wt0=np.ones(kernel_size[0:2], dtype=float)
        if taper:
            for ii in [0, -1]:
                wt0[ii, :] /= 2
                wt0[:, ii] /= 2
        wt0 = wt0.ravel()

        if lag is None:
            delta_subs=[di.ravel(), dj.ravel(), np.zeros_like(di.ravel())]
            wt=wt0
            grid_shift=[0, 0, 0]
        else:
            delta_subs=[
                np.concatenate([di.ravel(), di.ravel()]),
                np.concatenate([dj.ravel(), dj.ravel()]),
                np.concatenate([np.zeros_like(di.ravel(), dtype=int), np.zeros_like(di.ravel(), dtype=int)+lag])]
            wt = np.concatenate([-wt0, wt0])/(lag*self.grid.delta[2])
            grid_shift[2] = 0.5*lag*self.grid.delta[2]

        self.diff_op( delta_subs, wt.astype(float), which_nodes = ind0,\
                     valid_equations_only=valid_equations_only )
        if taper:
            self.update_dst_grid(grid_shift, kernel_size-1)
        else:
            self.update_dst_grid(grid_shift, kernel_size)

        return self

    def update_dst_grid(self, grid_shift, kernel_size):
        rcv0 = np.unravel_index(self.ind0-self.grid.col_0, self.grid.shape)
        # make a destination grid that spans the output data
        # the grid centers are shifted by grid_shift in each dimension
        dims=range(len(self.grid.shape))
        self.dst_grid = fd_grid(\
            [ [ self.grid.ctrs[dim][rcv0[dim][jj]] + grid_shift[dim] for jj in [0, -1]] for dim in dims],\
            kernel_size*self.grid.delta, name=self.name)
        # map the dst_ind0 value in the output grid
        out_subs = [ ((rcv0[dim]-rcv0[dim][0])/kernel_size[dim]).astype(int) for dim in dims ]
        self.dst_ind0 = np.ravel_multi_index( out_subs, self.dst_grid.shape)

        return self

    def data_bias(self, ind, val=None, col=None):
        # make a linear operator that returns a particular model parameter.
        # can be used to add one model parameter to a set of other parameters,
        # when added to another matrix, or to force a model parameter towards
        # a particular value, when appended to another matrix
        if col is None:
            col=self.col_N
            self.col_N +=1
        self.r=ind
        self.c=np.zeros_like(ind, dtype='int')+col
        if val is None:
            self.v=np.ones_like(ind, dtype='float')
        else:
            self.v=val.ravel()
        self.TOC['rows']={self.name:np.unique(self.r)}
        self.TOC['cols']={self.name:np.unique(self.c)}
        self.N_eq=np.max(ind)+1
        self.__update_size_and_shape__()
        return self

    def grid_prod(self, m, grid=None):
        # dot an operator with a vector, map the result to a grid
        if grid is None:
            if self.dst_grid is None:
                grid=self.grid
            else:
                grid=self.dst_grid
        if self.dst_ind0 is None:
            ind0=self.ind0
        else:
            ind0=self.dst_ind0
        P=np.zeros(grid.col_N+1)+np.NaN
        P[ind0]=self.toCSR(row_N=ind0.size).dot(m).ravel()
        return P[grid.col_0:grid.col_N].reshape(grid.shape)

    def grid_error(self, Rinv, grid=None):
        # calculate the error estimate for an operator and map the result to a grid
        if grid is None:
            if self.dst_grid is None:
                grid=self.grid
            else:
                grid=self.dst_grid
        if self.dst_ind0 is None:
            ind0=self.ind0
        else:
            ind0=self.dst_ind0
        E=np.zeros(self.col_N)+np.NaN
        E[ind0]=np.sqrt((self.toCSR(row_N=ind0.size).dot(Rinv)).power(2).sum(axis=1)).ravel()
        return E[grid.col_0:grid.col_N].reshape(grid.shape)

    def vstack(self, ops, order=None, name=None, TOC_cols=None):
        # combine a set of operators by stacking them vertically to form
        # a composite operator.  This could also be done by converting
        # the operators to sparse matrices and stacking them using the
        # vstack function, but using the lin_op.vstack() keeps track of
        # the different equations in the TOC
        if isinstance(ops, lin_op):
            ops=(self, ops)
        if order is None:
            order=range(len(ops))
        if name is not None:
            self.name=name
        if TOC_cols is None:
            TOC_cols=dict()
            col_array=np.array([], dtype=int)
            for op in ops:
                for key in op.TOC['cols'].keys():
                    TOC_cols[key]=op.TOC['cols'][key]
                    col_array=np.append(col_array, op.TOC['cols'][key])
            # add an entry for this entire operator
            if self.name is not None:
                TOC_cols[self.name]=np.unique(col_array)
        if self.col_N is None:
            self.col_N=np.max(np.array([op.col_N for op in ops]))

        self.TOC['cols']=TOC_cols
        # rr, cc, and vv are lists that will be populated with the nonzero
        # entries for each matrix being combined.
        rr=list()
        cc=list()
        vv=list()
        last_row=0
        for ind in order:
            # append the nonzero entries to the list of entries
            rr.append(ops[ind].r.ravel()+last_row)
            cc.append(ops[ind].c.ravel())
            vv.append(ops[ind].v.ravel())
            # label these equations in the TOC
            this_name=ops[ind].name
            if this_name is None:
                this_name='eq'
            # if TOC already contains the name we've specified, add a number to
            # distinguish this
            temp_name=this_name
            name_suffix=0
            while temp_name in self.TOC['rows'].keys():
                name_suffix+=1
                temp_name="%s_%d" %(this_name, name_suffix)
            if name_suffix>0:
                this_name="%s_%d" %(this_name, name_suffix)
            # shift the TOC entries and keep track of what sub-operators make up the current operator
            this_row_list=list()
            for key in ops[ind].TOC['rows'].keys():
                these_rows=np.array(ops[ind].TOC['rows'][key], dtype='int')+last_row
                self.TOC['rows'][key]=these_rows
                this_row_list.append(these_rows.ravel())
            # add a TOC entry for all of the sub operators together, if it's
            # not there already (which happens if we're concatenating composite operators)
            if this_name not in self.TOC['rows']:
                self.TOC['rows'][this_name]=np.concatenate(this_row_list)
            last_row+=ops[ind].N_eq
        # Combine the nonzero entries
        self.N_eq=last_row
        self.r=np.concatenate(rr)
        self.c=np.concatenate(cc)
        self.v=np.concatenate(vv)

        self.ind0=np.concatenate([op.ind0 for op in ops])
        if self.name is not None and len(self.name) >0:
            self.TOC['rows'][self.name]=np.arange(0, last_row)
        self.__update_size_and_shape__()
        return self

    def mask_for_ind0(self, mask_scale=None):
        """
        Sample the mask at the central indices for a linear operator

        This function samples the linear operator's mask field at the indices
        corresponding to the 'ind0' for each row of the operator.  The only
        input is:
             mask_scale (dict, or none):   gives the mapping between key values
                  and output values:  if mask_scale={1:0, 2:1}, then all mask
                  values equal to 1 will be returned as zero, and all mask values
                  equal to 2 will be returned as 1.
        """

        if self.grid.mask is None:
            return np.ones_like(self.ind0, dtype=float)
        if len(self.grid.shape) > len(self.grid.mask.shape):
            temp=np.unravel_index(self.ind0-self.grid.col_0, self.grid.shape)
            subs=tuple([temp[ii] for ii in range(len(self.grid.mask.shape))])
        else:
            inds=self.ind0-self.grid.col_0
            subs=np.unravel_index(inds, self.grid.mask.shape)
        temp=self.grid.mask[subs]
        if mask_scale is not None:
            temp2=np.zeros_like(temp)
            for key in mask_scale.keys():
                temp2[temp==key]=mask_scale[key]
            return temp2
        else:
            return temp

    def apply_2d_mask(self, mask=None):
        # multiply array elements by the values in a mask
        # The mask must have dimensions equal to the first two dimensions of
        # self.grid
        # if no mask is specified, use self.grid.mask
        if mask is None:
            mask=self.grid.mask
        csr=self.toCSR()

        for row in range(csr.shape[0]):
            # get the indices of the nonzero entries for the row
            inds=csr[row,:].nonzero()[1]
            # get the mask subscripts for the indices
            subs=np.unravel_index(inds-self.grid.col_0, self.grid.shape)
            # query the mask at those points
            mask_ind=np.ravel_multi_index([subs[0], subs[1]], mask.shape)
            # want to do: csr[row,inds] *= mask.ravel()[mask_ind]
            # but need to add a toarray() step to avoid broadcasting rules
            temp = csr[row, inds].toarray()
            csr[row,inds] = temp.ravel()*mask.ravel()[mask_ind]
        temp=csr.tocoo()
        self.r, self.c, self.v=[temp.row, temp.col, temp.data]
        return self

    def print_TOC(self):
        for rc in ('cols','rows'):
            print(rc)
            rc_min={k:np.min(self.TOC[rc][k]) for k in self.TOC[rc].keys()}
            for key in sorted(rc_min, key=rc_min.get):
                print("\t%s\t%d : %d" % (key, np.min(self.TOC[rc][key]), np.max(self.TOC[rc][key])))

    def fix_dtypes(self):
        self.r=self.r.astype(int)
        self.c=self.c.astype(int)

    def toCSR(self, col_N=None, row_N=None):
        # transform a linear operator to a sparse CSR matrix
        if col_N is None:
            col_N=self.col_N
        self.fix_dtypes()
        good=self.v.ravel() != 0
        if row_N is None:
            row_N=np.max(self.r.ravel()[good])+1
        return sp.csr_matrix((self.v.ravel()[good],(self.r.ravel()[good], self.c.ravel()[good])), shape=(row_N, col_N))
