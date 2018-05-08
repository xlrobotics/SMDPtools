from DFA import DFA
from DFA import Action
from MDP import MDP

if __name__ == '__main__':

    # DFA generated from LTL:
    # (F (G1 and G3) or F (G3 and G1)) and G !G2
    g1 = Action('g1')
    g2 = Action('g2') # set as an globally unsafe obstacle
    g3 = Action('g3')

    phi = Action('phi')

    dfa = DFA(0, [g1, g2, g3])

    dfa.set_final(3)
    dfa.set_sink(4)

    dfa.add_transition(phi.display(), 0, 0)
    dfa.add_transition(phi.display(), 1, 1)
    dfa.add_transition(phi.display(), 2, 2)
    dfa.add_transition(phi.display(), 3, 3)
    dfa.add_transition(phi.display(), 4, 4)

    dfa.add_transition(g1.display(), 0, 1)
    dfa.add_transition(g3.display(), 1, 3)
    dfa.add_transition(g3.display(), 0, 2)
    dfa.add_transition(g1.display(), 2, 3)

    dfa.add_transition(g1.display(), 1, 1)
    dfa.add_transition(g3.display(), 2, 2)

    dfa.add_transition(g1.display(), 3, 3)
    dfa.add_transition(g3.display(), 3, 3)

    dfa.add_transition(g1.display(), 4, 4)
    dfa.add_transition(g3.display(), 4, 4)

    dfa.add_transition(g2.display(), 0, 4)
    dfa.add_transition(g2.display(), 1, 4)
    dfa.add_transition(g2.display(), 2, 4)
    dfa.add_transition(g2.display(), 3, 4)
    dfa.add_transition(g2.display(), 4, 4)

    dfa.toDot("DFA")
    dfa.prune_eff_transition()

    print dfa.states
    print dfa.initial_state
    print dfa.sink_states
    print dfa.final_states
    print dfa.state_transitions
    print dfa.alphabet

    # s_q: dict {"state as index":"logic specification (transition in DFA) as content"}
    # q_s: dict {"logic specification (transition in DFA) as index":"state as content"}
    s_q = {}
    q_s = {g1.v:[], g2.v:[], g3.v:[], phi.v:[]}
    for i in range (1, 7):
        for j in range(1, 9):
            if (i, j) == (1, 2):
                s_q[(i, j)] = g1
                q_s[g1.v].append((i, j))
            elif (i, j) == (3, 4) or (i, j) == (4, 4) or (i, j) == (5, 4):
                s_q[(i, j)] = g2
                q_s[g2.v].append((i, j))
            elif (i, j) == (5, 7):
                s_q[(i, j)] = g3
                q_s[g3.v].append((i, j))
            else:
                s_q[(i, j)] = phi
                q_s[phi.v].append((i, j))

    # initialize origin MDP
    mdp = MDP()
    inner = []

    # define a 6*8 gridworld
    for i in range(1, 7):
        for j in range(1, 9):
            inner.append([i, j])

    mdp.set_S(inner)
    mdp.set_WallCord(mdp.add_wall(inner))
    mdp.set_P()
    mdp.set_L(s_q)
    mdp.set_Exp(q_s)

    print mdp.L
    # mdp.set_goal((2,1))
    # mdp.set_goal(())
    # mdp.init_V()
    # mdp.plot("test")
    # result = None
    result = mdp.product(dfa, mdp)
    # result.SVI(0.0000001)

    result.segmentation()
    result.svi_plot_key = True
    result.SVI(0.0000001)
    # result.layer_plot()

    result.SVI_option(0.0000001)
    # result.layer_plot()
    # result.plot_curve(result.action_special, result.hybrid_special, "compare_V_3_3")
    print result.hybrid_special






