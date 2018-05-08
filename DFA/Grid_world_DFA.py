from DFA import DFA
from DFA import Action
from MDP import MDP

if __name__ == '__main__':

    # DFA generated from LTL: Eventually visit g3 and g4 while avoiding r5
    r5 = Action('r5')
    g3 = Action('g3')
    g4 = Action('g4')
    phi = Action('phi')
    # ny = Action('g3', False)

    dfa = DFA(0, [r5, g3, g4])  # 0,['r', 'y', 'E']

    # print dfa.alphabet

    dfa.set_final(3)
    dfa.set_sink(4)

    # dfa.add_transition(y.conjunction(nr), 0, 1)
    # dfa.add_transition(nr.conjunction(ny), 0, 0)
    dfa.add_transition(phi.display(), 0, 0)
    dfa.add_transition(phi.display(), 1, 1)
    dfa.add_transition(phi.display(), 2, 2)
    dfa.add_transition(phi.display(), 3, 3)
    dfa.add_transition(phi.display(), 4, 4)

    dfa.add_transition(g3.display(), 0, 1)
    dfa.add_transition(g4.display(), 1, 3)
    dfa.add_transition(g4.display(), 0, 2)
    dfa.add_transition(g3.display(), 2, 3)

    dfa.add_transition(g3.display(), 1, 1)
    dfa.add_transition(g4.display(), 2, 2)

    dfa.add_transition(g3.display(), 3, 3)
    dfa.add_transition(g4.display(), 3, 3)

    dfa.add_transition(g3.display(), 4, 4)
    dfa.add_transition(g4.display(), 4, 4)

    dfa.add_transition(r5.display(), 0, 4)
    dfa.add_transition(r5.display(), 1, 4)
    dfa.add_transition(r5.display(), 2, 4)
    dfa.add_transition(r5.display(), 3, 4)
    dfa.add_transition(r5.display(), 4, 4)


    dfa.toDot("DFA")



    print dfa.states
    print dfa.initial_state
    print dfa.sink_states
    print dfa.final_states
    print dfa.state_transitions
    print dfa.alphabet

    s_q = {(2, 1):g3, (1, 3):g4, (2, 2):r5, (1, 1):phi, (1,2):phi, (2,3):phi}

    mdp = MDP()
    inner = []
    for i in range(1, 3):
        for j in range(1, 4):
            inner.append([i, j])

    mdp.set_S(inner)
    mdp.set_WallCord(mdp.add_wall(inner))
    mdp.set_P()
    mdp.set_L(s_q)

    print mdp.L
    # mdp.set_goal((2,1))
    # mdp.set_goal(())
    # mdp.init_V()
    # mdp.plot("test")
    mdp.product(dfa, mdp)


    # print mdp.P




