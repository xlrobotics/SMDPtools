from DFA import DFA
from DTS import DTS
from DFA import Action
from Queue import Queue

def crossproduct(a, b):
    return [(x, y) for x in a for y in b]


class product:
    def __init__(self):
        self.initial_state = None
        self.SQ = None
        # self.SQ_next = {}
        self.visited = []
        self.Act = None
        self.Delta = {}
        self.Iq0 = None
        self.SF = None

        self.ap_act = {}

    def set_ap_map(self, ap, act):
        if ap not in self.ap_act:
            self.ap_act[ap] = {}
        if act not in self.ap_act[ap]:
            self.ap_act[ap][act] = 1

        return

    def product(self, DFA, DTS):
        self.Iq0 = (DFA.initial_state, DTS.s_0)

        # self.Delta[self.Iq0] = DFA.initial_state

        self.SQ = crossproduct(DFA.states, DTS.states)
        self.Act = DTS.AP.keys()

        self.SF = crossproduct(DFA.final_states, DTS.states)

        self.Delta[tuple([self.Iq0[0]] + DTS.labels[self.Iq0[1]])] = self.delta(DFA.initial_state, DTS.labels[DTS.s_0], DFA.state_transitions)

        self.initial_state = (DFA.initial_state, DTS.s_0)

        current = Queue(0.1)

        current.put(self.initial_state)

        accepted = False

        print "Run:"
        count  = 1
        while not current.empty():
            sq = current.get()

            if sq in self.SF:
                accepted = True

            print "step - ", count, ':', sq
            count += 1
            for next_s in DTS.transitions[sq[1]]:
                 state = self.delta(sq[0], DTS.labels[next_s], DFA.state_transitions)

                 if state >= 0: # reachable
                     temp_node = tuple([state, next_s])

                     if temp_node not in self.visited:
                         self.visited.append(temp_node)
                         current.put(temp_node)

                         self.Delta[tuple([sq[0]] + DTS.labels[next_s])] = state
                     else:
                         pass


        # run from initial state to verify

        print 'The result of product run is:', accepted
        print 'Delta functions:', self.Delta
        print 'Combined initial state:', self.initial_state
        print 'All combined states:', self.SQ
        print 'Termination states:', self.SF

        self.visited = []

    def delta(self, q, labels, transitions):

        if len(labels) == 1:
            l = labels[0]
        else:
            l = tuple(labels)

        for index in transitions.keys():
            # print self.ap_act[l], labels, index[0], l

            if q == index[1] and index[0] in self.ap_act[l]:
                return transitions[index]
        return -1

    # def toDot(self):
    #     return






if __name__ == '__main__':

    r = Action('r')
    nr = Action('r', False)
    y = Action('y')
    ny = Action('y', False)

    dfa = DFA(0, [r, y]) #0,['r', 'y', 'E']

    # print dfa.alphabet

    dfa.set_final(2)

    # dfa.add_transition(y.conjunction(nr), 0, 1)
    # dfa.add_transition(nr.conjunction(ny), 0, 0)

    dfa.add_transition( (y.display(), nr.display()), 0, 1)
    dfa.add_transition( (nr.display(), ny.display()), 0, 0)

    dfa.add_transition(y.display(), 1, 1)
    dfa.add_transition(ny.display(), 1, 0)
    dfa.add_transition(r.display(), 0, 2)

    # print dfa.state_transitions

    dts = DTS('Sg')

    dts.add_states('Sy')
    dts.add_states('Sr')
    dts.add_states('Sry')
    dts.add_states('Sg')

    dts.add_labels('Sy', 'y')
    dts.add_labels('Sr', 'r')
    dts.add_labels('Sg', 'phi')
    dts.add_labels('Sry', 'phi')

    dts.add_transitions('Sg', 'Sy')
    dts.add_transitions('Sy', 'Sr')
    dts.add_transitions('Sr', 'Sry')
    dts.add_transitions('Sry', 'Sg')


    p = product()

    p.set_ap_map(('y'), ('y'))
    p.set_ap_map(('y'), ('y','!r'))
    p.set_ap_map(('y'), ('!r'))

    p.set_ap_map(('r'), ('r'))
    p.set_ap_map(('r'), ('r', '!y'))
    p.set_ap_map(('r'), ('!y'))

    p.set_ap_map(('phi'), ('!r', '!y'))

    p.product(dfa, dts)

    dfa.toDot("DFA struct")