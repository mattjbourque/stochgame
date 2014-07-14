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
Provides tools for working with two player stochastic games.  Most important
are the class StochGame for representing such games and the function
policy_iteration for solving them using an algorithm due to Raghavan and Syed
for the discounted case and Bourque and Raghavan in the average case.
"""
import stochmatrix as _st
import numpy as _np
import multilist as _ml
from copy import deepcopy

Strategy = _ml.Strategy

### Exceptions

class Error(Exception):
    """Base class for this module."""
    
    pass

class CycleError(Error):
    """Exception raised when policy iteration has cycled."""

    pass

##########################
# Utility functions for this module
##########################

def equal(x,y, tol=1e-6):
    """Returns True if |x - y| < tol"""
    
    x = float(x)
    y = float(y)
    
    if abs(x - y) < tol:
        return True
    else:
        return False

def pure_strategy_list(p1_num_actions, p2_num_actions):
    """
    Takes two lists of the same length representing the number of actions
    available in each state for the players.  Returns a list of tuples of the
    form (a1,a2,s), where a1 and a2 are actions for the two players and s is
    the state.

    The list is ordered like counting, but with the order of significant digits
    reversed - player 1's action is the least significant digit, the player 2's
    action, then the state.  (This is to match the way actions are listed in
    Gambit's text file format.)

    The main purpose of this function is for use in the constructor for the
    StochGame class.
    """

    if len(p1_num_actions) == len(p2_num_actions):
        n = len(p1_num_actions)
    else:
        raise ValueError("""Both players must have at least one action in all
        states.""")


    L = [(a1,a2,s) for s in range(n) 
                   for a2 in range(p2_num_actions[s])
                   for a1 in range(p1_num_actions[s])]
    
    return L

######


class StochGame(object):
    """
    StochGame(p1_num_actions,p2_num_actions,data)

    Returns a StochGame object representing a two player stochastic game.`

    Parameters
    ----------
    p1_num_actions : list-like object containing the number of actions
        available for player 1 in each state (length of the list is the number
        of states in the game.)
    p2_num_actions : same as above, for player 2.  Note that these two lists 
        must be of the same length.
    data : a list-like object.  Each element in the list gives the rewards to
        each player and the transition vector when players use a particular
        pair of actions in a state.  The elements are taken to be ordered with
        player 1's action least significantly, then player 2's action, and the
        state most significantly.  Each element e should be a list-like object of
        length 3: e[0] is player 1's reward, e[1] is player 2's reward, and
        e[2] is a tuple of length n (number of states) giving a probability
        vector (the transition).
    
    See also
    --------
    read_stochgame_file

    Examples
    --------
    The two state, nonzero sum game

        +----------+   +-----------+----------+
        |         1|   |          5|         7|
        | (1.0, 0) |   | (.5, .5)  | (.8, .2) |
        |0         |   |4          |6         |
        +----------+   +-----------+----------+
        |         3|
        | (0, 1.0) |
        |2         |
        +----------+

    (where the upper right number is player 2's payoff and the lower left is
    player 1's payoff with the transition vector in the middle) would be
    entered as

    >>> G = StochGame([2, 1], [1, 2], [(0, 1, [1.0, 0.0]), (2, 3, [0.0, 1.0]),\\
                                     (4, 5, [0.5, 0.5]), (6, 7, [0.8, 0.2])])

    """

    def __init__(self, p1_num_actions,p2_num_actions, data):

        if len(p1_num_actions) == len(p2_num_actions):
            self.num_states = len(p1_num_actions)
            self.num_actions =\
                    _ml.MultiList([p1_num_actions,p2_num_actions],'p1','p2')
        else:
            raise ValueError("""Both players must have at least one action in each state""") 
        
        data = list(data)
       
        self.statelist = range(self.num_states)

        action_pair_record = _np.dtype([ ('p1', 'float32'),
                                         ('p2', 'float32'),
                                         ('t', 'float64', (self.num_states,)) ])
        shape = [(p1_num_actions[state], p2_num_actions[state]) for state in self.statelist]

        self.data = [_np.zeros(shape[state], dtype=action_pair_record) for state in self.statelist]


        strategies = pure_strategy_list(self.num_actions['p1'],
                self.num_actions['p2'])

        for pair in strategies:

            state = pair[2]
            actions = pair[:2]

            self.data[state][actions] = data.pop(0)



    def __repr__(self):

        L = []
        for state in range(self.num_states): 
            for datum in self.data[state].flat:
                L.append(datum)

        out = "StochGame("
        out += str(self.num_actions['p1']) + ", " \
            + str(self.num_actions['p2']) + ", "
        out += str(L)
        out +=  ")"

        return out

    def transition(self, state, s1, s2):
        """
        transition(state, s1, s2)
        
        Returns a transition vector in the form of a 1 x self.num_states matrix
        for players using actions 's1' and 's2' in state 'state'.

        Parameters
        ----------
        state : an integer in self.statelist
        s1 : an positive integer < self.num_actions['p1'] representing an
            action for player 1
        s2 : an positive integer < self.num_actions['p2'] representing an
            action for player 2

        See Also
        --------
        transitionmatrix

        Examples
        --------

        >>> G = StochGame([2 1], [1 2], [(0, 1, [1.0, 0.0]), (2, 3, [0.0, 1.0]),\\
                                         (4, 5, [0.5, 0.5]), (6, 7, [0.8, 0.2])])

        >>> G.transition(0,0,0)
        matrix([[ 1.,  0.]])

        """
        
        return _np.matrix(self.data[state][s1,s2]['t'])


    def __str__(self):
        """
        Produces a string compatible with the read_stochgame_file function, so
        that it can be used for writing games to a file.

        See Also
        --------
        read_stochgame_file

        Examples
        --------
        >>> G = StochGame([2,1], [1,2], [(0,1,[1.0,0.0]), (2,3,[0.0,1.0]),\\
                    (4,5,[0.5,0.5]), (6,7,(0.8,0.2)) ])
        >>> f = open("game.stg","w")
        >>> f.write(G.__str__())
        >>> f.close()
        >>> H = read_stochgame_file("game.stg"); print H
        StochGame([2 1],  ... ) # same as G

        """

        out = str(self.num_actions['p1']) + " # player 1 number of actions\n"
        out += str(self.num_actions['p2']) + " # player 2 number of actions\n\n"
        
        for state in range(self.num_states):
            out += "# state " + str(state) + "\n"
            row = 0
            col = 0
            for datum in self.data[state].flat:

                if row == self.num_actions['p1'][state]:
                    row = 0
                    col += 1
                
                if row == 0:
                    out += "# action %s for player 2\n" % col

                out += str(datum) + "# action %s for player 1\n" % row
                row += 1

            out += "\n"

        return out


    def transitionmatrix(self, strategy):
        """
        transitionmatrix(strategy)
        
        Returns a stochastic transition matrix (class stomat) corresponding
        to the given (pure stationary) strategy.

        Parameters
        ----------
        strategy: an object of type multilist.Strategy giving a pure stationary
            strategy pair.

        See Also
        --------
        transition

        Examples
        --------
        >>> G = StochGame([2,1], [1,2], [(0,1,[1.0,0.0]), (2,3,[0.0,1.0]),\\
                    (4,5,[0.5,0.5]), (6,7,(0.8,0.2)) ])
        >>> s = multilist.Strategy((0,0),(0,0)); G.transition(s)
        stomat([[ 1. ,  0. ],
           [ 0.5,  0.5]])

        """

        L = []
        for state in self.statelist:
            L += list(self.data[state][ strategy[state] ]['t'])
    
        temp = _np.matrix(L).reshape((self.num_states,self.num_states))

        return _st.stomat(temp)

    def reward(self, strategy):
        """
        reward(strategy)

        Returns an outcome object giving the reward to the players for using a
        pure stationary strategy.  In terms of typical notation, if the
        strategy pair is (f,g), this returns an object representing the vectors
        r_1(f,g) and r_2(f,g).

        Parameters
        ----------
        strategy: an object of type multilist.Strategy giving a pure stationary
            strategy pair.

        Examples
        --------
        >>> G = StochGame([2,1], [1,2], [(0,1,[1.0,0.0]), (2,3,[0.0,1.0]),\\
                    (4,5,[0.5,0.5]), (6,7,(0.8,0.2)) ])
        >>> s = multilist.Strategy((0,0),(0,0)); G.reward(s)
        Outcome([(0.0, 1.0), (4.0, 5.0)], 
          dtype=[('p1', '<f4'), ('p2', '<f4')])

        """
        
        rv1 = [self.data[state][ strategy[state] ]['p1'] for state in self.statelist]
        rv2 = [self.data[state][ strategy[state] ]['p2'] for state in self.statelist]
        
        return _ml.Outcome((rv1,rv2))

    def payoff(self, strategy, discount=None):
        """
        payoff(strategy, discount=None)

        Returns an object of type multilist.Outcome representing discounted or
        limiting average payoff for the two players using a given pure
        stationary strategy.  In typical notation, for pure stationary strategy
        pair (f,g), returns x(f,g) (limiting average) or v_{beta}(f,g)
        (discounted).
        
        Parameters
        ----------
        strategy: an object of type multilist.Strategy giving a pure stationary
            strategy pair.
        discount : a float at least zero and less than 1, or 'None'. If 'None',
            the limiting average payoff is returned.

        See Also
        --------
        bias, zee

        Examples
        --------
        >>> G = StochGame([2,1], [1,2], [(0,1,[1.0,0.0]), (2,3,[0.0,1.0]),\\
                    (4,5,[0.5,0.5]), (6,7,(0.8,0.2)) ])
        >>> s = multilist.Strategy((0,0),(0,0))
        >>> G.payoff(s)
        Outcome([(0.0, 1.0), (0.0, 1.0)], 
          dtype=[('p1', '<f4'), ('p2', '<f4')])

        >>> G.payoff(s, discount=0.8)
        Outcome([(-3.7025504190379976e-16, 5.0),
           (6.666666507720947, 11.666666984558105)], 
           dtype=[('p1', '<f4'), ('p2', '<f4')])

        """
        r = self.reward(strategy)

        if discount == None:
            Q = self.transitionmatrix(strategy).lim

            return Q*r

        elif 0<= discount < 1:

            M = (_np.identity(self.num_states) -
                    discount*self.transitionmatrix(strategy)).I
            return M*r
        else:
            raise ValueError("Discount factor must be between zero and one.")

    def bias(self, strategy):
        """
        bias(strategy)

        Returns 'bias vector' for a given pure stationary strategy pair.  The
        bias for a pss pair (f,g) is denoted 'y(f,g)' in [Blackwell 1962] and
        [Veinott 1964].

        See Also
        --------
        payoff, zee

        """

        D = self.transitionmatrix(strategy).D
        r = self.reward(strategy)

        return D*r

    def zee(self, strategy):
        """
        zee(strategy)

        Returns 'z vector' for a given pure stationary strategy pair.  The
        bias for a pss pair (f,g) is denoted 'z(f,g)' in [Veinott 1964].

        See Also
        --------
        payoff, bias

        """
        
        D = self.transitionmatrix(strategy).D
        r = self.reward(strategy)

        return D**2*(-1)*r

    def belonging_to(self, player):
        """
        belonging_to(player)

        Returns a list of states belonging to "player".

        Parameters
        ----------
        player : A string, either 'p1' or 'p2' representing player 1 or player
        2.
        """

        ret = []

        for state in self.statelist:
            if self.num_actions[player][state] > 1:
                ret.append(state)

        return ret

    def strategy_pair_generator(self):
        """
        Returns a generator which iterates over all strategy pairs.
        """
        
        N = self.num_states
        current = Strategy([[0]*N]*2)
        p1done = False

        while not p1done:
            
            p2done = False

            while not p2done:

                yield current

                p2state = N-1

                while p2state >= 0 and current['p2'][p2state] \
                        == self.num_actions['p2'][p2state] -1:

                            current['p2'][p2state] = 0
                            p2state -= 1

                if p2state >= 0:
                    current['p2'][p2state] += 1

                if p2state < 0:
                    p2done = True

            p1state = N-1
            while p1state >=0 and current['p1'][p1state] \
                    == self.num_actions['p1'][p1state] - 1:

                        current['p1'][p1state] = 0
                        p1state -=1

            if p1state >= 0:
                current['p1'][p1state] += 1

            if p1state < 0:
                p1done = True

class MDP(StochGame):
    """
    Defines a MDP class representing stochastic games in which at most one
    player has a choice of actions available.
    """

    def __init__(self, num_actions, data):

        num_states = len(num_actions)
        dummy = [1]*num_states

        StochGame.__init__(self, num_actions,dummy,data)

    def __repr__(self):

        L = []
        for state in range(self.num_states): 
            for datum in self.data[state].flat:
                L.append(datum)

        out = "MDP("
        out += str(self.num_actions['p1']) + ", " \
            + str(self.num_actions['p2']) + ", "
        out += str(L)
        out +=  ")"

        return out
        
        
    def transition(self, state, s):
        return StochGame.transition(self,state,s,0)


    def transitionmatrix(self, strategy):
        
        s = sg.Strategy(strategy,[0]*self.num_states)
        return StochGame.transitionmatrix(self, s)

    def reward(self, strategy):
        
        s = sg.Strategy(strategy,[0]*self.num_states)
        return StochGame.reward(self, s)

    def payoff(self, strategy, discount=None):

        s = sg.Strategy(strategy,[0]*self.num_states)
        return StochGame.payoff(self, s, discount)['p1']

    def bias(self, strategy):

        s = sg.Strategy(strategy,[0]*self.num_states)
        return StochGame.bias(self, s)['p1']

    def zee(self, strategy):

        s = sg.Strategy(strategy,[0]*self.num_states)
        return StochGame.zee(self, s)['p1']


def read_stochgame_file(filename):
    """ 
    Reads a stochastic game from a text file.

    Parameters
    ----------
    filename : A stochastic game file: a text file with the following syntax:

        - All blank lines or lines beginning with # (comments) are ignored.
    
        - The first two lines read of the form
          [N1, N2, ..., Nn]
          [M1, M2, ..., Mn]

          which give the number of actions in each state for player 1 and
          player 2 respectively.  

        - Each remaining line read is of the form
          (r1,r2,(p1, p2, ..., pn))

          where r1 is a reward to player 1, r2 is a reward to player 2, and
          (p1,..,pn) is a transition vector.  These are listed in the same
          ordering as read by the StochGame constructor: player 1's action is
          least significant, the player 2's action, and the state is most
          significant.

    Examples
    --------

    file game.stg
    +------------------------------------------+
    | [2, 1] # player 1 number of actions      |
    | [1, 2] # player 2 number of actions      |
    |                                          |
    |# state 0                                 |
    |# action 0 for player 2                   |
    |(2, 3, [0.0, 1.0])# action 1 for player 1 |
    |                                          |
    |# state 1                                 |
    |# action 0 for player 2                   |
    |(4, 5, [0.5, 0.5])# action 0 for player 1 |
    |# action 1 for player 2                   |
    |(6, 7, [0.8, 0.2])# action 0 for player 1 |
    |                                          |
    +------------------------------------------+

    >>> H = read_stochgame_file("game.stg")
    >>> print H
    [2, 1] # player 1 number of actions
    [1, 2] # player 2 number of actions
    ...
    # action 1 for player 2
    (6, 7, [0.8, 0.2])# action 0 for player 1

    """
    
    f = open(filename)
    
    got_actions = 0
    line_n0 = 0
    L = []

    for line in f:
        line = line.split("#")[0]
        line = line.strip()
        if line != "":
            if got_actions <1:
                exec("p1_num_actions = " + line)
                got_actions += 1
            elif got_actions == 1:
                exec("p2_num_actions = " + line)
                got_actions += 1
            else:
                exec("L.append(" + line +")")

    return StochGame(p1_num_actions,p2_num_actions,L)



    f.close()

    return StochGame(D)

def adj_improve_discount(G, current, discount, minimize=False):
    """
    Checks for an adjacent improvement using the discount improvement criteria
    based on [Blackwell 1962].  Returns an improvement if there is one.  If
    there is not, simply returns 'current'.

    Parameters
    ----------
    G : a StochGame object representing a zero sum game of perfect information
    current : a multilist.Strategy object giving a pure stationary strategy for the
        players in G
    discount : a discount factor at least zero and less than 1
    minimize: if True, looks for an improvement for the minimizer

    See Also
    --------
    adj_improve_average, PolicyIterator, policy_iteration

    Note
    ----
    This algorithm only works for perfect information zero sum games, but the code does
    not currently check whether G has this property.

    """

    if minimize:
        the_player = 'p2'
    else:
        the_player = 'p1'
    
    next_s = deepcopy(current)

    # x and y are as defined in Blackwell:
    # x(f) = Q(f)*r(f)
    # y(f) = H(f)*r(f), where H(f) is the deviation matrix
    
    x_curr = G.payoff(current, discount)[the_player]

    for state in G.belonging_to(the_player):
        a = 0
        b = 0
        while [a,b][0 if the_player=='p1' else 1] < G.num_actions[the_player][state]:
            if a == current['p1'][state] and the_player == 'p1':
                a += 1
            elif b == current['p2'][state] and the_player == 'p2':
                b += 1
            else:
                p = G.transition(state,a,b)
                r = G.data[state][(a,b)][the_player]

                new = r + discount*(p*x_curr)
                old = x_curr[state]
                
                tolerance = min([1e-6, (1-discount)/1000])

                if (not equal(new,old,tolerance) and new > old ):
                    if the_player == 'p1':
                        next_s[the_player][state] = a
                    elif the_player == 'p2':
                        next_s[the_player][state] = b
                    return next_s
                else:
                    if the_player == 'p1':
                        a += 1
                    elif the_player == 'p2':
                        b += 1

    return next_s

def adj_improve_average(G, current, minimize=False, returnE=False):
    """
    Checks for a (non-adjacent) improvement using the average improvement criteria
    based on [Blackwell 1962].  If there is an improvement, returns it alone if
    returnE is False, or a tuple consisting of the improved strategy and None
    if returnE is True.  If there is not an improved strategy and returnE is
    False, returns the existing strategy; if returnE is true, gives a tupe of
    the existing strategy and a list of lists of actions in each state for
    which both of Blackwell's criteria hold with equality.

    returnE should be True when this is called as a first step for finding
    1-optimal strategies with adj_improve_one.

    Parameters
    ----------
    G : a StochGame object representing a zero sum game of perfect information.
    current : a multilist.Strategy object giving a pure stationary strategy for the
        players in G
    minimize: if True, looks for an improvement for the minimizer

    See Also
    --------
    adj_improve_discount, PolicyIterator, policy_iteration
    
    Note
    ----
    This algorithm only works for perfect information zero sum games, but the code does
    not currently check whether G has this property.

    """

    if minimize:
        the_player = 'p2'
    else:
        the_player = 'p1'

    next_s = deepcopy(current)
    E = [[] for i in range(G.num_states)]
    
    # x and y are as defined in Blackwell:
    # x(f) = Q(f)*r(f), where Q(f) is the Cesaro limit matrix
    # y(f) = H(f)*r(f), where H(f) is the deviation matrix
    
    x_curr = G.payoff(current)[the_player]

    for state in G.belonging_to(the_player):
        a = 0 if the_player=='p2' else current['p1'][state]
        b = 0 if the_player=='p1' else current['p2'][state]
        while [a,b][0 if the_player=='p1' else 1] < G.num_actions[the_player][state]:
            if a == current['p1'][state] and the_player == 'p1':
                a += 1
            elif b == current['p2'][state] and the_player  == 'p2':
                b += 1
            
            else:
                p = G.data[state][a,b]['t']
                r = G.data[state][a,b][the_player]

                new1 = p*x_curr
                old1 = x_curr[state]

                if (not equal(new1,old1) and new1 > old1 ):
                    if the_player == 'p1':
                        next_s[the_player][state] = a
                    if the_player == 'p2':
                        next_s[the_player][state] = b

                    break
                    
#                     if returnE:
#                         return next_s, None
#                     else:
#                         return next_s

                elif equal(new1, old1):

                    y_curr = G.bias(current)[the_player]
                    new2 = r + p*y_curr
                    old2 = x_curr[state] + y_curr[state]

                    if (not equal(new2, old2) and new2 > old2):
                        if the_player == 'p1':
                            next_s[the_player][state] = a
                        if the_player == 'p2':
                            next_s[the_player][state] = b

                        break

#                         if returnE:
#                             return next_s, None
#                         else:
#                             return next_s
                    
                    else:
                        if equal(new2, old2):
                            E[state].append( [a,b][0 if the_player=='p1' else 1] )
                        if the_player == 'p1':
                            a += 1
                        elif  the_player  == 'p2':
                            b += 1
                else:
                    if the_player == 'p1':
                        a += 1
                    elif the_player  == 'p2':
                        b += 1

    if returnE:
        return next_s, E
    else:
        return next_s

def adj_improve_one(G, current, E, minimize=False):
    """
    Checks for a (non-adjacent) improvement using the improvement criteria
    based on [Veinott 1965].  If there is an improvement, returns a tuple
    consisting of the improved strategy and None.  If there is not an improved
    strategy, returns the existing strategy and a list of lists of actions in
    each state for which both of Blackwell's criteria hold with equality.

    Parameters
    ----------
    G : a StochGame object representing a zero sum game of perfect information.
    current : a multilist.Strategy object giving a pure stationary strategy for the
        players in G
    E : A list of lists: E[s] is a list of actions in state s which satisfy
        Blackwell's average improvement criteria with equality.
    minimize: if True, looks for an improvement for the minimizer

    See Also
    --------
    adj_improve_discount, PolicyIterator, policy_iteration
    
    Note
    ----
    This algorithm only works for perfect information zero sum games, but the code does
    not currently check whether G has this property.

    """

    if minimize:
        the_player = 'p2'
    else:
        the_player = 'p1'

    next_s = deepcopy(current)
    num_E = [len(E[i]) for i in range(len(E))]
    
    # z is as defined in Veinott:
    # z(f) = H(f)*(-y(f)), where H(f) is the deviation matrix
    
    y_curr = G.bias(current)[the_player]
    z_curr = G.zee(current)[the_player]
    
    for state in G.belonging_to(the_player):
        a = current['p1'][state]
        b = current['p2'][state]
        while E[state]:
            if the_player == 'p1':
                a = E[state].pop()
                b = 0
            elif the_player == 'p2':
                b = 0
                a = E[state].pop()

            p = G.data[state][a,b]['t']
            new = p*z_curr
            old = y_curr[state] + z_curr[state]

            if (not equal(new, old) and new > old):
                if the_player == 'p1':
                    next_s[the_player][state] = a
                if the_player == 'p2':
                    next_s[the_player][state] = b

                #return next_s
                break
    return next_s
                
class PolicyIterator(object):
    """
    PolicyIterator(G, start, discount=None, check_one_opt=True)
    
    Returns an iterator object which will return improved strategies in a
    perfect information zero sum game until none is available, when it raises a
    StopIteration.  Will always return an improvement for player 1 when one is
    available, and will return an improvement for player 2 only if there's none
    for player 1.

    Parameters
    ----------
    G : a StochGame object representing a zero sum game of perfect information
    start : A multilist.Strategy object representing a pure stationary strategy
        pair for the two players.
    discount : Either 'None' or a discount factor at least zero and less than
        one.  If 'None', checks for limiting average improvements.  If a
        discount factor, checks for discounted improvements.
    check_one_opt: (Boolean) Should we guarantee a one-optimal policy for
        player 1 at each iteration using Veinott's algorithm?  Proof of the
        algorithm seems to require this step, but at least some games are
        successfully solved without it.

    See Also
    --------
    adj_improve_average, adj_improve_discount

    Examples
    --------
    >>> G =  StochGame([2 1], [1 2], [(0, 1, [1.0, 0.0]), (2, 3, [0.0, 1.0]),\\
                        (4, 5, [0.5, 0.5]), (6, 7, [0.8, 0.2])])
    >>> s = multilist.Strategy((0,0), (0,0))
    >>> pi = PolicyIterator(G,s)
    >>> t = pi.next();print t
    [(1, 0), (0, 0)]

    >>> G.payoff(s)['p1']; G.payoff(t)['p1'] # t is an improvement for player 1
    Outcome([ 0.,  0.], dtype=float32)
    Outcome([ 3.33333325,  3.33333325], dtype=float32)

    Notes
    -----
    The algorithm implemented is only guaranteed to work on zero-sum two-player
    games with additive rewards and additive transitions, but we do not check
    these properties of the game.  If the game is not of this class, the
    algorithm may cycle.  However, in the event that it does terminate, the
    strategy pair is optimal, even outside of the class of ARAT games.
    """


    def __init__(self, G, start, discount = None, check_one_opt=True):

        self.game = G
        self.current = deepcopy(start)
        self.discount = discount
        self.E1 = None
        self.E2 = None
        self.last_was_degen=False
        self.check_one_opt = check_one_opt

    def next(self):

        self.last_was_degen=False

        if self.discount == None:
            
            old = _ml.Strategy(self.current)
            self.current, self.E1 =\
                adj_improve_average(self.game,self.current,minimize=False,returnE=True)

            if self.current == old and self.E1 != None and self.check_one_opt:
                self.current = adj_improve_one(self.game,self.current,
                        self.E1,minimize=False)
                if self.current != old:
                    self.last_was_degen = True
            
            if self.current == old:
                self.current, self.E2 =\
                    adj_improve_average(self.game,self.current,minimize=True,returnE=True)

            else:
                return self.current

            if self.current == old:
                raise StopIteration
            else:
                return self.current

        elif 0 <= self.discount < 1:
            old = _ml.Strategy(self.current)
            self.current =\
                adj_improve_discount(self.game,self.current,
                        discount = self.discount,minimize=False)

            if self.current == old:
                self.current =\
                    adj_improve_discount(self.game,self.current,
                            discount = self.discount, minimize=True)
            else:
                return self.current

            if self.current == old:
                raise StopIteration
            else:
                return self.current

        else:
            raise ValueError("""Discount factor must be between zero and
            one.""")

def policy_iteration(G,start=None,discount=None,verbose=True, \
        showboth=False, check_one_opt=True,decimals=4):
    """
    Solves a two player zero sum stochastic game of perfect information. If
    'verbose' is True, prints intermediate strategies and returns a tuple of
    the optimal strategy and the total number of iterations required.  If
    'verbose' is false, prints no intermediate information and returns only the
    optimal strategy (in the form of a multilist.Strategy object).

    Parameters
    ----------
    G : A StochGame object representing a two player zero sum game of perfect
        information
    start : A multilist.Strategy object representing a pure stationary strategy; the
        starting place for the policy iteration algorithm.  If not provided,
        the strategy pair consisting of the first action listed for each player
        in each state is assumed; that is, Strategy( [[0]*G.num_states]*2 ).
    discount : Either 'None' or a discount factor at least zero and less than
        one.  If 'None', solves the game using the limiting average criterion.
        Otherwise uses the discounted payoff criterion.
    verbose : boolean.  Print progress information?
    showboth : boolean.  If verbose, show both players' payoffs?
    check_one_opt: (Boolean) Should we guarantee a one-optimal policy for
        player 1 at each iteration using Veinott's algorithm?  Proof of the
        algorithm seems to require this step, but at least some games are
        successfully solved without it.
    decimals : If verbose, how many decimals to round payoffs to?

    See Also
    --------
    adj_improve_discount, adj_improve_average

    Examples
    --------
    >>> G =  StochGame([2 1], [1 2], [(0, 1, [1.0, 0.0]), (2, 3, [0.0, 1.0]),\\
                        (4, 5, [0.5, 0.5]), (6, 7, [0.8, 0.2])])
    >>> s = multilist.Strategy((0,0), (0,0))
    >>> policy_iteration(G,s)

    >>> optimal,k = policy_iteration(G,s) 
    [1 0] [0 0] [ 3.33333325  3.33333325]
    [1 0] [0 1] [ 4.22222233  4.22222233]

    >>> print optimal; G.payoff(optimal)['p1']
    [(1, 0) (0, 1)]
    Outcome([ 4.22222233,  4.22222233], dtype=float32)

    """
    #total_strategies = reduce(lambda x,y:x*y,
    #        G.num_actions['p1']+G.num_actions['p2'])

    if start == None:
        start = [[0]*G.num_states]*2
    start =  Strategy(start)

    pi = PolicyIterator(G, deepcopy(start), discount, check_one_opt)
    iterations = 0
    degen = 0
    
    s = start
    visited = [s]

    if verbose:
        report = str(s['p1']) + str(s['p2']) \
                + G.payoff(s,discount).pretty(decimals, showboth)

        if discount==None:
            report += G.bias(s).pretty(decimals, showboth) \
                + G.zee(s).pretty(decimals, showboth)

        print report

    while True:
        try:
            iterations += 1
            s = pi.next()
            if pi.last_was_degen:
                degen += 1
            if s in visited:
                print s
                raise CycleError
            else:
                visited.append(s)
        except StopIteration:
            break
        
        if verbose:
            report = str(s['p1']) + str(s['p2']) \
                    + G.payoff(s,discount).pretty(decimals, showboth)

            if discount==None:
                report += G.bias(s).pretty(decimals, showboth) \
                    + G.zee(s).pretty(decimals, showboth)
                if pi.last_was_degen:
                    report += "*"

            print report
    
    return s,pi.E1,pi.E2,iterations, degen

def gametable(G):
    """
    Returns a table for all the pure strategy pairs available in the game G.
    """

    print "header"

    for pair in pure_strategy_list(G.num_actions['p1'], G.num_actions['p2']):
        print pair


def multi_improve_average(G, current, minimize=False, returnE=False):
    """
    Checks for an (non-adjacent) improvement using the average improvement criteria
    based on [Blackwell 1962].  If there is an improvement, returns it alone if
    returnE is False, or a tuple consisting of the improved strategy and None
    if returnE is True.  If there is not an improved strategy and returnE is
    False, returns the existing strategy; if returnE is true, gives a tupe of
    the existing strategy and a list of lists of actions in each state for
    which both of Blackwell's criteria hold with equality.

    returnE should be True when this is called as a first step for finding
    1-optimal strategies with adj_improve_one.

    Parameters
    ----------
    G : a StochGame object representing a zero sum game of perfect information.
    current : a multilist.Strategy object giving a pure stationary strategy for the
        players in G
    minimize: if True, looks for an improvement for the minimizer

    See Also
    --------
    adj_improve_discount, PolicyIterator, policy_iteration
    
    Note
    ----
    This algorithm only works for perfect information zero sum games, but the code does
    not currently check whether G has this property.

    """

    if minimize:
        the_player = 'p2'
    else:
        the_player = 'p1'

    next_s = deepcopy(current)
    E = [[] for i in range(G.num_states)]
    
    # x and y are as defined in Blackwell:
    # x(f) = Q(f)*r(f), where Q(f) is the Cesaro limit matrix
    # y(f) = H(f)*r(f), where H(f) is the deviation matrix
    
    x_curr = G.payoff(current)[the_player]

    for state in G.belonging_to(the_player):
        a = 0 if the_player=='p2' else current['p1'][state]
        b = 0 if the_player=='p1' else current['p2'][state]
        while [a,b][0 if the_player=='p1' else 1] < G.num_actions[the_player][state]:
            if a == current['p1'][state] and the_player == 'p1':
                a += 1
            elif b == current['p2'][state] and the_player  == 'p2':
                b += 1
            
            else:
                p = G.data[state][a,b]['t']
                r = G.data[state][a,b][the_player]

                new1 = p*x_curr
                old1 = x_curr[state]

                if (not equal(new1,old1) and new1 > old1 ):
                    if the_player == 'p1':
                        next_s[the_player][state] = a
                    if the_player == 'p2':
                        next_s[the_player][state] = b
                    
                    break

                elif equal(new1, old1):

                    y_curr = G.bias(current)[the_player]
                    new2 = r + p*y_curr
                    old2 = x_curr[state] + y_curr[state]

                    if (not equal(new2, old2) and new2 > old2):
                        if the_player == 'p1':
                            next_s[the_player][state] = a
                        if the_player == 'p2':
                            next_s[the_player][state] = b

                        break

                    else:
                        if equal(new2, old2):
                            E[state].append( [a,b][0 if the_player=='p1' else 1] )
                        if the_player == 'p1':
                            a += 1
                        elif  the_player  == 'p2':
                            b += 1
                else:
                    if the_player == 'p1':
                        a += 1
                    elif the_player  == 'p2':
                        b += 1

    if returnE:
        return next_s, E
    else:
        return next_s
