__author__ = 'Xuan Liu, xliu9@wpi.edu'

import numpy as np

class DTS:
    def __init__(self, init_state = None, states = None, labels = None, action = None, goals = None, AP = None):
        if init_state is not None:
            self.s_0 = init_state
        else:
            self.s_0 = 0

        self.states = set([])
        # self.final_states = set([])
        self.transitions = {}
        self.action = {}
        self.labels = {}
        self.AP = {"phi":[True,False]}


    def add_AP(self, ap):
        if ap not in self.AP:
            self.AP[ap] = [True, False]

    def add_labels(self, state, label):
        if state not in self.labels:
            self.labels[state] = []
            self.add_AP(label)
        self.labels[state].append(label)

    def add_states(self, state):
        self.states.add(state)

    def add_transitions(self, current, next):
        if current not in self.transitions:
            self.transitions[current] = []
        self.transitions[current].append(next)

    def set_s0(self, state):
        self.s_0 = state

    def L(self, state):

        if state not in self.labels:
            return ""
        else:
            return self.labels[state]



    # def

if __name__ == '__main__':
    dts = DTS('Sg')

    dts.add_states('Sy')
    dts.add_states('Sr')
    dts.add_states('Sry')
    dts.add_states('Sg')

    dts.add_labels('Sy', 'y')
    dts.add_labels('Sr', 'r')

    dts.add_transitions('Sg', 'Sy')
    dts.add_transitions('Sy', 'Sr')
    dts.add_transitions('Sr', 'Sry')
    dts.add_transitions('Sry', 'Sg')

    # dts.add_AP('r')
    # dts.add_AP('y')


    print dts.labels
    print dts.transitions
    print dts.AP

    # for element in dts.states:
    #     print element, dts.L(element)
