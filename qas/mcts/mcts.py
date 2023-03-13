from copy import deepcopy
import random
from typing import (
    List,
    Sequence,
    Callable,
    Optional,
    Union,
    Set
)
import numpy as np
from qas.mcts.mcts_state import StateOfMCTS
class TreeNode():
    # https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
    def __init__(self, state:StateOfMCTS, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __repr__(self):
        return "State: {}\nParent: {}\nNum Visits: {}\nTotal Reward: {}\nChildern: {}\nisTerminal: {}\nisFullyExpanded: {}".format(self.state, self.parent, self.numVisits, self.totalReward, self.children, self.isTerminal, self.isFullyExpanded)
