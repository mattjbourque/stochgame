# Stochgame - a Python module for stochastic games
# Copyright (C) 2012 Matthew Bourque
# 
# This file is part of Stochgame.
# 
# Stochgame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Stochgame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Stochgame.  If not, see <http://www.gnu.org/licenses/>.
# 
# For questions about Stochgame, contact:
# 
#     Matthew Bourque (mbourque@math.uic.edu)
#     www.math.uic.edu/~mbourque/stochgame/
#     Department of Mathematics, Statistics, and Computer Science
#     University of Illinois at Chicago
#     322 Science and Engineering Offices (M/C 249)
#     851 S. Morgan St.
#     Chicago IL 60607-7045

"""
A utility module to provide tools for working with stochastic matrices.
"""
## todo
# - fix cesaro_limit to use ergodic_classes function, rather than "ergodicize"

# from numpy import matrix, zeros_like, sum, delete, zeros, ones, identity,\
#asmatrix

import numpy as np

from numpy.linalg import solve

# Utility functions for this module

def is_stochastic(P, tol=.1e-5):
    """Returns True if P is a stochastic matrix, False otherwise.  Here,
    "stochastic matrix" means the rows sums are 1 +- tol, and all entries 
    are nonzero.
    """

    # If it's not square, it't not a stochastic matrix.
    if P.shape[0] != P.shape[1]:
        return False

    # If it has negative entries, it's not a stochastic matrix.
    # This is not a good check.  Something like -8e-18 should pass.
    #if True in (P < zeros_like(P)):
    #    return False

    # If any rows don't sum to one, its not a stochastic matrix.
    rowsums = np.sum(P, axis = 1).A1.tolist()
    for row in range(len(rowsums)):
        # magic comparison!
        if abs(rowsums[row] - 1) > tol:
            return False

    # Otherwise, its OK.
    return True

def zero_oneify(P):
    """
    Accepts an array P and turns it into a zero-one matrix P' where P'_{ij}=0
    if P_{ij} = 0, and P'_{ij}=1 otherwise.  

    Warning: this will change the original matrix as well as returning the
    new zero-one version.  Maybe that should not be the case.
    """
    (rows, cols) = P.shape
    for i in range(rows):
        for j in range(cols):
            if P[i,j] > 0: P[i,j] = 1

    return P

def permutation( perm ):
    """
    Returns a permutation matrix that sends range(len(perm))  to perm.
    """

    N = len(perm)

    P = np.zeros( (N, N) )

    for i in range(N):
        P[i, perm[i]] = 1

    return P

def indices_of(sought, roworcol):
    """ 
    Accepts <roworcol> (a 1xN or Nx1 matrix row or column) and returns
    a set of indices of items that are in <sought>.
    """
        
    # if it's a column, make it a row
    if roworcol.shape[1] == 1:
        row = roworcol.T
    else:
        row = roworcol

    N = row.shape[1]

    return set( ind for ind in range(N) if row[0,ind] in sought )

def submatrix(P, rowstates, colstates):
    """
    returns the submatrix of P consisting of only rows from rowstates and
    columns from colstates
    """

    nrow= len(rowstates)
    ncol = len(colstates)

    row_ind = list( rowstates[i] for i in range(nrow) for j in range(ncol) )
    col_ind = list( colstates[j] for i in range(nrow) for j in range(ncol) )

    return P[row_ind, col_ind].reshape( (nrow,ncol) )

def ergodicize(P):
    """
    Takes a stochastic matrix P and returns an StateSetsMatrix with
    the first state set containing all transient states of P and
    the remaining state sets each corresponding to an ergodic class.
    
    Implements algorithm in Fox and Landi 1968

    Probably shouldn't be called other than by the function ergodic_classes.
    """

    ssP = StateSetMatrix(np.matrix(P.__array__()))
    
    # Step 1 - mark absorbing states as ergodic and mark any states which 
    # have access to them as transient
    for state in range(ssP.num_state_sets):
        is_absorbing = ssP.check_absorbing(state)
        if is_absorbing:
            for trans_state in ssP.go_to(state):
                ssP.update_label(trans_state, "transient")

    while len(ssP.unlabeled) > 0:
        unlabeled_states = iter(ssP.unlabeled)
        state = unlabeled_states.next()
        chain = []

        while state in ssP.unlabeled and state not in chain:
            chain.append( state )
            state = iter( ssP.come_from(state) ).next()

        # If we left the while loop because we found a state labeled
        # transient, then the whole chain must be transient.
        if ssP.label[state] == "transient":
            for trans_state in chain:
                ssP.update_label(trans_state,"transient")

        # If we left the while loop because we came back to the chain, we have
        # found a set of communicating states, and we collapse them and check
        # whether the new state set is absorbing.  If so, we label the state
        # set ergodic.  If not, we leave it unlabeled for now.
        elif state in chain:
            ind = chain.index(state)
            del chain[0:ind]
            
            # collapse into the smallest-indexed state set, so that we check
            # this new state set for absorbingness
            state = min(chain)
            ssP.collapse_state_sets(state, chain)
            is_absorbing = ssP.check_absorbing(state)
            if is_absorbing:
                for trans_state in ssP.go_to(state):
                    ssP.update_label(trans_state, "transient")

        
    # Finally, we collapse all transient states
    ssP.collect_label("transient")


    return ssP

def ergodic_classes(P):
    """accepts a matrix P (presumably stochastic, but not checked) and 
    returns a list [c1, c2, ..., cn, bool] where c1, c2, ..., cn are lists
    of the states of each ergodic class of P, and bool=True if the final 
    state is transient.
    """
    
    cmP = ergodicize(P)

    ret = list(cmP.state_set)
    if "transient"in cmP.label:
        has_transient = True
        ret.append(True)
    else:
        has_transient = False
        ret.append(False)

    return ret

def cesaro_limit(P, ecP=None):
    """Returns the Cesaro limit of a matrix P.  If also passed a StateSetMatrix
    giving the ergodic classes of P, it will use that instead of calculating the
    ergodic classes from scratch.  Should probably be rewritten to accept only
    the list of ergodic classes, rather than the whole StateSetMatrix.

    Intended as a function to be used by the property lim of stomat class.
    """

    Q = np.matrix(P.__array__()) 
    N = Q.shape[0]
    Qstar = np.zeros_like(Q)


    if ecP:
        ecQ = ecP

    else:
        # Change this to use ergodic_classes.  It's not necessary to carry around
        # the whole state set matrix.
        ecQ = ergodicize(Q)

    if ecQ.label[-1] == "transient":
        erg_cls_indices = range(ecQ.num_state_sets)[:-1]
        trans_cls = ecQ.state_set[-1]
    else:
        erg_cls_indices = range(ecQ.num_state_sets)
        trans_cls=False

    # for each ergodic class, calculate the stationary vector, repeat it to
    # form a matrix, and put it into Qstar
    for ind in erg_cls_indices:
        erg_cls = ecQ.state_set[ind]
        q = submatrix(Q, erg_cls, erg_cls)
        n = q.shape[0]
        J = np.ones( (n,n) ) - np.identity(n)
        b = np.ones( (n,1) )
        pi = solve( q.T + J, b)
        qstar = np.matrix( pi.T.tolist()*n )

        row_ind = list(erg_cls[i] for i in range(n) for j in range(n))
        row_ind = np.matrix(row_ind).reshape( (n,n) )

        col_ind = list(erg_cls[j] for i in range(n) for j in range(n))
        col_ind = np.matrix(col_ind).reshape( (n,n) )
        Qstar[ row_ind, col_ind ] = qstar

        # now calculate the stationary probabilities conditional on starting in
        # each tranient state, using formula from Filar's appendix
        if trans_cls:
           n_trans = len(trans_cls) 
           qtrans = submatrix(Q, trans_cls, trans_cls)
           qtranserg = submatrix(Q, trans_cls, erg_cls)

           # qtrans is the matrix P_{L+1} in the appendix - it is the transition
           # matrix for the transient subchain.  qtranserg is the matrix 
           # P_{L+1, l} from Filar - the transition matrix from transient classes
           # to the ergodic class l ("ell")

           qstar_te = (np.identity(n_trans) - qtrans).I*qtranserg*qstar

           row_ind = list(trans_cls[i] for i in range(n_trans) for j in range(n))
           row_ind = np.matrix(row_ind).reshape( (n_trans, n) )

           col_ind = list(erg_cls[j] for i in range(n_trans) for j in range(n) )
           col_ind = np.matrix(col_ind).reshape(n_trans, n)

           Qstar[ row_ind, col_ind] = qstar_te
        

    return stomat(Qstar)

# end of utilities


class StateSetMatrix:
    """class StateSetMatrix
    
    A class for holding useful information about a transition matrix.
    Probably not very useful directly; it's intended as a tool for the 
    ergodicize function, in turn for the ergodic_classes function.
    """

    def __init__(self,P):
        N = P.shape[0]
        self.matrix = zero_oneify( P.copy() )
        self.num_state_sets = N

        #each state in this matrix corresponds to a *set* of classes in the original
        self.state_set = list([i] for i in range(N) )
        
        self.label = list("unlabeled" for i in range(N) )
        self.unlabeled = list( ind for ind in range(N) if self.label[ind] == "unlabeled" )
    
    def __str__(self):
        out = "There are " + str(self.num_state_sets) + " state sets \n"
        for i in range(self.num_state_sets):
            out = out + str(self.state_set[i]) + " is " + self.label[i] + ".\n"

        return out
    
    def update_unlabeled(self):
        N = self.num_state_sets
        self.unlabeled = list( ind for ind in range(N) if self.label[ind] == "unlabeled" )

    def collapse_state_sets(self, remain, gonerlist ):
        
        newP = self.matrix

        #don't include <remain> in <gonerlist>
        if remain in gonerlist: gonerlist.remove(remain)
        
        # This loop does two things:
        # - update the matrix by adding rows from <gonerlist> to row <remain>
        #   and adding columns from <gonerlist> to column <remain>
        # - update state sets: put state sets in gonerlist into the set <remain>
        for goner in gonerlist:

            # update state sets
            self.state_set[remain] = self.state_set[remain] + self.state_set[goner]

            # update rows but don't delete old ones yet
            newP[remain, :] = newP[remain, :] + newP[goner, :]

            # update columns but don't delete old ones yet
            newP[:, remain] = newP[:, remain] + newP[:, goner]


        # N is the old number of state sets - we'll loop over the old set of 
        # unlabled state sets and the list of state set labels, and keep only the ones
        # not in <gonerlist>
        N = self.num_state_sets

        # Keep only state sets and labels not in <gonerlist>
        self.state_set = list( self.state_set[ind] for ind in range(N) \
                if ind not in gonerlist )
        self.label = list( self.label[ind] for ind in range(N) \
                if ind not in gonerlist )

        # delete the rows and columns of the matrix and state sets in <gonerlist>
        newP = np.delete(newP, gonerlist, 0)
        newP = np.delete(newP, gonerlist, 1)

        self.matrix = zero_oneify(newP)

        # update number of state sets
        newN = self.num_state_sets = newP.shape[0]

        # remake the list of unlabeled state sets
        self.update_unlabeled()

    def check_absorbing(self, state):

        row = self.matrix[state,:]
        diag = self.matrix[state, state]

        if diag == 1 and len( indices_of([1], row) ) == 1:
            self.label[state] = "ergodic"
            self.update_unlabeled()
            return True
        else:
            return False

    def come_from(self, state, include_self=False):
        
        if include_self:
            return indices_of([1], self.matrix[state,:] )
        else:
            return indices_of([1], self.matrix[state,:] ).difference( [state] )

    def go_to(self, state, include_self=False):

        if include_self:
            return indices_of([1], self.matrix[:,state] )
        else:
            return indices_of([1], self.matrix[:,state] ).difference( [state] )

    def update_label(self, state, label):

        self.label[state] = label
        self.update_unlabeled()
    
    def reorder_indices(self, neworder):
        
        # Should complain here.  For now silently do nothing
        if len(neworder) != self.num_state_sets:
            return

        self.state_set = list( self.state_set[neworder[i]] 
                for i in range(self.num_state_sets) )
        self.label = list( self.label[neworder[i]] 
                for i in range(self.num_state_sets) )
        
        self.update_unlabeled()

        # permute matrix rows and columns
        Perm = permutation( neworder )
        self.matrix = Perm.T*self.matrix*Perm
        

    def collect_label(self, label):

        if label not in self.label:
            return

        # Collapse all state sets with <label> into the first such one

        current_ind = self.label.index(label)
        gonerlist = list( ind for ind in range(self.num_state_sets) \
                if self.label[ind] == label)
        self.collapse_state_sets(current_ind, gonerlist)
        
        neworder = range(self.num_state_sets)
        neworder.remove(current_ind)

        self.reorder_indices( neworder + [current_ind] )

class stomat(np.matrix):
    """ Defines a stochastic matrix class.  """

#     def __init__(self,data,dtype="float64",copy=False):
#         matrix.__init__(self, data, dtype, copy)

    def __new__(subtype, data, dtype="float64", copy=True):

        if is_stochastic(np.asmatrix(data)):

            if isinstance(data, np.matrix):
                data = np.asarray(data)
                return np.matrix.__new__(subtype,data,dtype,copy)

            else:
                return np.matrix.__new__(subtype,data,dtype,copy)

        else:
            return np.matrix.__new__(np.matrix,data,dtype,copy)

    def __array_finalize__(self,obj):

        np.matrix.__array_finalize__(self,obj)

        self._classes = None
        self._limit = None
        self._deviation = None
        
    def __repr__(self):
        s = repr(self.__array__()).replace('array', 'stomat')
        return s

    def __mul__(self,other):

        prod = np.matrix.__mul__(self,other)
        
        if isinstance(other,stomat):
            return prod.view(stomat)
        else:
            return prod.view(np.matrix)

    def __rmul__(self,other):

        prod = np.matrix.__rmul__(self, other)

        if isinstance(stomat):
            return prod.view(stomat)
        else:
            return prod.view(np.matrix)

    def __getitem__(self,index):

        out = np.matrix.__getitem__(self,index)

        if is_stochastic(out):
            return out
        else:
            return out.view(np.matrix)


    def __getslice__(self,i,j):
        return self.__getitem__(slice(i,j))

    def get_classes(self):
        
        if self._classes == None:
            self._classes = ergodic_classes(self)

        return self._classes

    def get_lim(self):

        if self._limit == None:
            ecssm = ergodicize(self)
            self._limit = cesaro_limit(self, ecssm)

        return self._limit

    def getD(self):

        if self._deviation == None:
            N = self.shape[0]
            self._deviation = (np.identity(N) - self.__array__() 
                    + self.lim).I - self.lim

        return self._deviation

    classes = property(get_classes, None, 
            doc = "ergodic classes list & boolean transient indicator")
    lim = property(get_lim, None, doc = "Cesaro limit")
    D = property(getD, None, doc = "deviation matrix")
            
            


