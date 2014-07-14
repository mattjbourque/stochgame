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
Provides functions for generating random games.
"""

from stochgame import StochGame
from numpy import around as _around
from numpy import array as _array
from numpy.random.mtrand import dirichlet as _dirichlet
from numpy.random import random_integers as _randint
from numpy.random import random as _randnum
from random import choice as _choice

def random_transition(n,k,a,decimals):
    """
    Returns an array of size n x k.  Each row is a k-dimensional probability
    vector (i.e. with positive entries summing to 1) drawn from a Dirichlet
    distribution with concentration parameter a.

    Parameters
    ----------

    n : Number of vectors to be drawn.

    k : Dimension of each vector.

    a : concentration parameter for the Dirichlet distribution.  Each vector
        will be drawn from a Dir([a]*k) distribution.

    decimals : A rounding parameter.  All entries will be rounded to this
               number of decimal places.
    """
    
    A =   _around(_dirichlet([a]*k, n), decimals)

    for row in range(n):
        A[row] = A[row]/sum(A[row])

    return A

def integer_rewards(rbound, N):
    """
    Returns an array of N integer rewards with absolute value <= rbound
    """
    
    return _randint(-rbound,rbound,N)

def float_rewards(rbound,N):
    """
    Returns an array of N float rewards with absolute value <= rbound
    """
    
    signs = _array([_choice([-1,1])]*N)

    return rbound*signs*_randnum(N)

def random_game(actions1, actions2, rbound = 5, integer_payoffs=True, a=0.2, decimals=2, zerosum=True):
    """
    A function for randomly generating two-player zero sum stochastic games with
    integer payoffs.

    Parameters
    ----------

    actions1 : list giving the number of actions for player 1 in each state

    actions2 : list giving the number of actions for player 2 in each state 

    rbound: Bound on absolute value of immediate rewards.

    integer_payoffs: boolean.  Should the payoffs be integers?

    a : concentration parameter passed to random_transition

    decimals : rounding parameter passed to random_transition

    zerosum : boolean.  If true, the returned game is zero-sum.

    """
    
    if len(actions1) != len(actions2):
        raise ValueError("Mismatch in number of states for two players.")
    else: 
        n = len(actions1)
        total_actions = sum( [actions1[i]*actions2[i] for i in range(n)] )

    data = []

    if integer_payoffs: 
        rewards =  integer_rewards
    else:
        rewards = float_rewards

    trans = random_transition(total_actions,n,a,decimals)
    r1 = rewards(rbound,total_actions)
    if zerosum:
        r2 = -r1
    else:
        r2 = rewards(rbound,total_actions)
    
    for pair in range(total_actions):
        data.append( (r1[pair], r2[pair], trans[pair].tolist() ) )
    
    return StochGame(actions1, actions2, data)

def random_arat(actions1, actions2, rbound = 5, p = .5, q= .5, integer_payoffs=True, a=0.2, decimals=2):
    """
    A function for randomly generating two-player zero sum ARAT stochastic games with
    integer payoffs.

    Parameters
    ----------

    actions1 : list giving the number of actions for player 1 in each state

    actions2 : list giving the number of actions for player 2 in each state 

    rbound: Bound on absolute value of immediate rewards.

    p,q: weights for two players' contributions. Normalized to sum to 1

    integer_payoffs: boolean.  Should the payoffs be integers?

    a : concentration parameter passed to random_transition

    decimals : rounding parameter passed to random_transition

    """
    
    if len(actions1) != len(actions2):
        raise ValueError("Mismatch in number of states for two players.")
    else:
        n = len(actions1)
        total_actions = sum(actions1) + sum(actions2)

    norm = float(p+q)
    p = p/norm
    q= q/norm

    data = []

    if integer_payoffs:
        rewards =  integer_rewards
    else:
        rewards = float_rewards

    for state in range(n):
        reward_rowpart = rewards(rbound,actions1[state])
        trans_rowpart = random_transition(actions1[state],n,a,decimals)
        reward_colpart = rewards(rbound,actions2[state])
        trans_colpart = random_transition(actions2[state],n,a,decimals)
        for column in range(actions2[state]):
            for row in range(actions1[state]):
                r = p*reward_rowpart[row] + q*reward_colpart[column]
                trans = p*trans_rowpart[row] + q*trans_colpart[column]
                data.append( (r, -r, trans.tolist() ) )
    
    return StochGame(actions1, actions2, data)
