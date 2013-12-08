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
Support for containers for pairs of vectors to support payoffs and other
outcome-type vector pairs in two-player games.

Classes
-------
MultiList

Strategy

Outcome

"""
import numpy as _np

class MultiList(dict):
    """
    A container for multiple list-like, same-length objects indexed by strings.
    Primarily intended as a base class for Strategy and Outcome classes.
    
    The intended way to call this is with a list of same-length lists L and
    len(L) strings:
    M = MultiList(L,s0,s2,...)-> multilist with M[si] = L[i]

    If its called with some other args, these will be passed to the builtin
    dict constructor.

    A multilist M can be indexed by its keys (strings) just like a dict.  It
    can also be indexed by an int k, which will return a tuple made up of the
    kth element of each list in the multilist.  The entries in the tuple will
    be ordered by sorting the keys of their corresponding lists.

    Note
    ----
    This base class does not do any checking regarding the contents of the
    list L of values.  This could lead to problems with the expanded indexing
    capabilities if subclasses aren't careful.
    """
    
    def __init__(self, *args, **kwargs):
        try: 
            if len(args[0])==len(args[1:]):
                dict.__init__(self,zip(args[1:],args[0]))
            else:
                dict.__init__(self,*args,**kwargs)
        except TypeError:
            dict.__init__(self,*args,**kwargs)

    def __str__(self):
        
        s = ""
        keys = self.keys(); keys.sort()

        for key in keys:
            s += str(key) + ": " + str(self[key]) + "\n"

        return s

    def __getitem__(self, key):

        if key in self.keys():
            return dict.__getitem__(self,key)
        else:
            keys = self.keys(); keys.sort()
            L = [self[k][key] for k in keys]
            return tuple(L)

class Strategy(MultiList):
    """
    A class for pure stationary strategies in two player games.

    Examples
    --------

    >>> s = Strategy([(0,0,0), (0,0,0)])
    >>> t = Strategy((0,0,0), (0,0,0))
    """

    def __init__(self,*data):

        if len(data) == 1:
            if type(data[0])==Strategy:
                    s = data[0]
                    data = [s['p1'], s['p2']]
                    MultiList.__init__(self,data,'p1','p2')
            else:
                data = [list(data[0][0]), list(data[0][1])]
                MultiList.__init__(self,data,'p1','p2')

        elif len(data) == 2:
                data = [list(data[0]), list(data[1])]
                MultiList.__init__(self,data,'p1','p2')
        
        else:
            raise ValueError('Strategies only for two players.')

class Outcome(MultiList):
    """
    A container for a pair of vectors.  Used to represent an 'outcome' in a
    two player stochastic game; that is x,y, or z vector as defined in
    [Veinott, 1966].

    If x and y are two outcomes:

    x + y -> Outcome(x['p1'] + y[p1], x['p2'] + y['p2'])

    M*x -> Outcome(M*x['p1'], M*x['p2'])
    """
    
    def __init__(self,data):

        if len(data) != 2:
            raise ValueError('Outcome only for two players.')
        
        if len(data[0]) != len(data[1]):
            raise ValueError('Vector length mismatch')
            
        if type(data)==Outcome:
            data = [ data['p1'], data['p2'] ]

        data = [_np.matrix(x).reshape((-1,1)) for x in data]

        MultiList.__init__(self,data,'p1','p2')

    def lists(self):
    
        data = [self[p].flatten().tolist() for p in ['p1','p2']]

        return MultiList(data, 'p1', 'p2')
    
    def __str__(self):

        return self.lists().__str__()
        
    
    def __add__(self,other):

        try:
            d1 = self['p1'] + other['p1']
            d2 = self['p2'] + other['p2']
            return Outcome([d1,d2])
        except TypeError:
            return other.__add__(self)


    def __rmul__(self,other):

        try:
            return Outcome( [other*self['p1'], other*self['p2']] )
        except TypeError:
            return other.__mult__(self)
    
    def pretty(self, decimals=4, showboth=False):
        """
        Returns a pretty string representation of the outcome.

        Parameters
        ----------
        decimals : round all entries to this many decimal places.

        showboth : boolean. If true, show payoffs for both players.  If false
                   (the default), show only the payoffs to player 1.

        """

        if not showboth:

            out = _np.around(self['p1'], decimals).flatten().tolist()

            return str(out)
        
        else:
            
            x = _np.around(self['p1'], decimals).flatten().tolist()
            y = _np.around(self['p2'], decimals).flatten().tolist()

            return str(zip(x,y))
