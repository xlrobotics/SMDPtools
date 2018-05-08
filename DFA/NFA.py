__author__ = 'Jie Fu, jfu2@wpi.edu'

from types import *

class ExceptionFSM(Exception):

    """This is the FSM Exception class."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

class NFA:

    """This is a module for Nondeterministic Finite State Automaton (NFA).
    """

    def __init__(self, initial_state=None, alphabet=None, transitions=dict([]) ,final_states=None, memory=None):
        # Initialize the automaton. The machine has all components that can be defined later.
        self.state_transitions = {}
        self.final_states = set([])
        self.state_transitions=transitions
        if alphabet == None:
            self.alphabet=[]
        else:
            self.alphabet=alphabet
        self.initial_state = initial_state
        if self.initial_state == None:
            pass
        else:
            self.states =[ initial_state ] # the list of states in the machine.
        
    def reset (self):
        """This sets the current_state to the initial_state and sets
        input_symbol to None. The initial state was set by the constructor
         __init__(). """
        self.current_state = self.initial_state
        self.input_symbol = None

    def add_transition(self,input_symbol,state,next_state_list):
        """
        state ---input_symbol ---> next_state_list
        using list instead of set for the set of next state to be reached.
        appending the state set if the next state was not in.
        """
        if state not in self.states:
            self.states.append(state)
        for each_next in next_state_list:
            if each_next not in self.states:
                self.states.append(each_next)
        if input_symbol not in self.alphabet:
            self.alphabet.append(input_symbol)
        if self.state_transitions.has_key((input_symbol,state)):
            pass
        else:
            self.state_transitions[input_symbol,state]=next_state_list
        return
    
    def get_transition (self, input_symbol, state):
        """
        This returns a list of next states given an input_symbol and state.
        """

        if self.state_transitions.has_key((input_symbol, state)):
            return self.state_transitions[(input_symbol, state)]
        else:
            return None
