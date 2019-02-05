# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:25:46 2017

@author: ben
"""
import numpy as np
import scipy.sparse as sp

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

    def diff_op(self, delta_subs, vals,  which_nodes=None):
        # build an operator that calculates linear combination of the surrounding
        # values at each node of a grid.
        # A template, given by delta_subs and vals contains a list of offsets
        # in each direction of the grid, and a list of values corresponding
        # to each offset.  Only those nodes for which the template falls
        # entirely inside the grid are included in the operator

        # compute the maximum and minimum offset in each dimension
        max_deltas=[np.max(delta_sub) for delta_sub in delta_subs]
        min_deltas=[np.min(delta_sub) for delta_sub in delta_subs]
        #generate the center-node indices for each calculation
        # if in dimension k, min_delta=-a and max_delta = +b, the number of indices is N,
        # then the first valid center is a and the last is N-b
        sub0s=np.meshgrid(*[np.arange(np.maximum(0, -min_delta), np.minimum(Ni, Ni-max_delta)) for Ni, min_delta, max_delta in zip(self.grid.shape, min_deltas, max_deltas)])
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
            self.c[:,ii]=self.grid.global_ind(this_sub)
            self.v[:,ii]=vals[ii]
        self.ind0=self.grid.global_ind(sub0s).ravel()
        self.TOC['rows']={self.name:range(self.N_eq)}
        self.TOC['cols']={self.grid.name:np.arange(self.grid.col_0, self.grid.col_0+self.grid.N_nodes)}
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
        return self

    def grad(self, DOF='z'):
        coeffs=np.array([-1., 1.])/(self.grid.delta[0])
        dzdx=lin_op(self.grid, name='d'+DOF+'_dx').diff_op(([0, 0],[-1, 0]), coeffs)
        dzdy=lin_op(self.grid, name='d'+DOF+'_dy').diff_op(([-1, 0],[0, 0]), coeffs)
        self.vstack((dzdx, dzdy))
        return self

    def grad_dzdt(self, DOF='z'):
        coeffs=np.array([-1., 1., 1., -1.])/(self.grid.delta[0]*self.grid.delta[2])
        d2zdxdt=lin_op(self.grid, name='d2'+DOF+'_dxdt').diff_op(([ 0, 0,  0, 0], [-1, 0, -1, 0], [-1, -1, 0, 0]), coeffs)
        d2zdydt=lin_op(self.grid, name='d2'+DOF+'_dydt').diff_op(([-1, 0, -1, 0], [ 0, 0,  0, 0], [-1, -1, 0, 0]), coeffs)
        self.vstack((d2zdxdt, d2zdydt))
        return self

    def diff(self, lag=1, dim=0):
        coeffs=np.array([-1., 1.])/(lag*self.grid.delta[dim])
        deltas=[[0, 0] for this_dim in range(self.grid.N_dims)]
        deltas[dim]=[0, lag]
        self.diff_op((deltas), coeffs)
        return self

    def dzdt(self, lag=1, DOF='dz'):
        coeffs=np.array([-1., 1.])/(lag*self.grid.delta[2])
        self.diff_op(([0, 0], [0, 0], [0, lag]), coeffs)
        return self

    def d2z_dt2(self, DOF='dz'):
        coeffs=np.array([-1, 2, -1])/(self.grid.delta[2]**2)
        self=lin_op(self.grid, name='d2'+DOF+'_dt2').diff_op(([0,0,0], [0,0,0], [-1, 0, 1]), coeffs)
        return self

    def grad2(self, DOF='z'):
        coeffs=np.array([-1., 2., -1.])/(self.grid.delta[0]**2)
        d2zdx2=lin_op(self.grid, name='d2'+DOF+'_dx2').diff_op(([0, 0, 0],[-1, 0, 1]), coeffs)
        d2zdy2=lin_op(self.grid, name='d2'+DOF+'_dy2').diff_op(([-1, 0, 1],[0, 0, 0]), coeffs)
        d2zdxdy=lin_op(self.grid, name='d2'+DOF+'_dxdy').diff_op(([-1, -1, 1,1],[-1, 1, -1, 1]), 0.5*np.array([-1., 1., 1., -1])/(self.grid.delta[0]**2))
        self.vstack((d2zdx2, d2zdy2, d2zdxdy))
        return self

    def grad2_dzdt(self, DOF='z'):
        coeffs=np.array([-1., 2., -1., 1., -2., 1.])/(self.grid.delta[0]**2.*self.grid.delta[2])
        d3zdx2dt=lin_op(self.grid, name='d3'+DOF+'_dx2dt').diff_op(([0, 0, 0, 0, 0, 0],[-1, 0, 1, -1, 0, 1], [-1,-1,-1, 0, 0, 0]), coeffs)
        d3zdy2dt=lin_op(self.grid, name='d3'+DOF+'_dy2dt').diff_op(([-1, 0, 1, -1, 0, 1], [0, 0, 0, 0, 0, 0], [-1, -1, -1, 0, 0, 0]), coeffs)
        coeffs=np.array([-1., 1., 1., -1., 1., -1., -1., 1.])/(self.grid.delta[0]**2*self.grid.delta[2])
        d3zdxdydt=lin_op(self.grid, name='d3'+DOF+'_dxdydt').diff_op(([-1, 0, -1, 0, -1, 0, -1, 0], [-1, -1, 0, 0, -1, -1, 0, 0], [-1, -1, -1, -1, 0, 0, 0, 0]),  coeffs)
        self.vstack((d3zdx2dt, d3zdy2dt, d3zdxdydt))
        return self

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
        return self

    def data_bias(self, ind, col=None):
        # make a linear operator that returns a particular model parameter.
        # can be used to add one model parameter to a set of other parameters,
        # when added to another matrix, or to force a model parameter towards
        # a particular value, when appended to another matrix
        if col is None:
            col=self.col_N
            self.col_N +=1
        self.r=ind
        self.c=np.zeros_like(ind, dtype='int')+col
        self.v=np.ones_like(ind, dtype='float')
        self.TOC['rows']={self.name:np.unique(self.r)}
        self.TOC['cols']={self.name:np.unique(self.c)}
        self.N_eq=np.max(ind)+1
        return self

    def grid_prod(self, m):
        # dot the operator with a vector, map the result to the operator's grid
        P=np.zeros(self.col_N)+np.NaN
        P[self.ind0]=self.toCSR().dot(m)
        return P[self.grid.col_0:self.grid.col_N].reshape(self.grid.shape)

    def grid_error(self, Rinv):
        # calculate the error estimate for an operator and map the result to the operator's grid
        E=np.zeros(self.col_N)+np.NaN
        E[self.ind0]=np.sqrt((self.toCSR().dot(Rinv)).power(2).sum(axis=1))
        return E[self.grid.col_0:self.grid.col_N].reshape(self.grid.shape)

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
            # add a TOC entry for all of the sub operators together
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
        return self

    def mask_for_ind0(self, mask_scale=None):
        if self.grid.mask is None:
            return np.zeros_like(self.ind0, dtype=float)
        if len(self.grid.shape) > len(self.grid.mask.shape):
            temp=np.unravel_index(self.ind0-self.grid.col_0, self.grid.shape)
            subs=[temp[ii] for ii in range(len(self.grid.mask.shape))]
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

    def print_TOC(self):
        for rc in ('cols','rows'):
            print(rc)
            rc_min={k:np.min(self.TOC[rc][k]) for k in self.TOC[rc].keys()}
            for key in sorted(rc_min, key=rc_min.get):
                print("\t%s\t%d : %d" % (key, np.min(self.TOC[rc][key]), np.max(self.TOC[rc][key])))

    def fix_dtypes(self):
        self.r=self.r.astype(int)
        self.c=self.c.astype(int)

    def toCSR(self, col_N=None):
        # transform a linear operator to a sparse CSR matrix
        if col_N is None:
            col_N=self.col_N
        self.fix_dtypes()
        return sp.csr_matrix((self.v.ravel(),(self.r.ravel(), self.c.ravel())), shape=(np.max(self.r.ravel())+1, col_N))
