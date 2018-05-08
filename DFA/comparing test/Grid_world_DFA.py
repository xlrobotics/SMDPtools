from DFA import DFA
from DFA import Action
from MDP import MDP

if __name__ == '__main__':

    # DFA generated from LTL:
    # (F (G1 and G3) or F (G3 and G1)) and G !G2
    g1 = Action('g1')
    g2 = Action('g2')  # set as an globally unsafe obstacle
    g3 = Action('g3')
    g23 = Action('g23')
    obs = Action('obs')
    phi = Action('phi')
    whole = Action('whole')

    row, col = 6, 8
    i, j = range(1, row + 1), range(1, col + 1)
    states = [[x, y] for x in i for y in j]
    S = [tuple([x, y]) for x in i for y in j]

    # print S

    s_q = {}
    q_s = {}
    q_s[g1.v] = [(3, 1), (3, 2)]  # (1, 2), (1, 3)
    q_s[obs.v] = [(1, 6), (2, 4), (2, 6), (3, 4), (4, 4), (5, 1), (5, 2), (6, 1), (6, 2),
                  (6, 6)]  # (3, 4), (4, 4), (5, 4)
    q_s[g2.v] = [(3, 8), (4, 8)]
    q_s[g23.v] = [(4, 8)]
    q_s[g3.v] = [(4, 8), (5, 8)]  # (5, 7), (6, 7)
    q_s[phi.v] = list(set(S) - set(q_s[g1.v] + q_s[g2.v] + q_s[g3.v] + q_s[obs.v]))
    q_s[whole.v] = S
    for s in S:
        if s in q_s[g1.v]:
            s_q[s] = g1.v
        elif s in q_s[g2.v]:
            s_q[s] = g2.v
        elif s in q_s[g3.v]:
            s_q[s] = g3.v
        elif s in q_s[g23.v]:
            s_q[s] = g23.v
        elif s in q_s[obs.v]:
            s_q[s] = obs.v
        else:
            s_q[s] = phi.v

    # initialize origin MDP
    mdp = MDP()
    mdp.set_S(states)
    mdp.set_WallCord(mdp.add_wall(states))
    mdp.set_P()
    mdp.set_L(s_q)
    mdp.set_Exp(q_s)
    # print "probabilities", len(mdp.P)

    dfa = DFA(0, [g1, g2, obs, g3, g23, phi, whole])

    dfa.set_final(4)
    dfa.set_sink(5)

    sink = list(dfa.sink_states)[0]

    for i in range(sink + 1):
        dfa.add_transition(phi.display(), i, i)
        if i < sink:
            dfa.add_transition(obs.display(), i, sink)

    dfa.add_transition(whole.display(), sink, sink)

    dfa.add_transition(g1.display(), 0, 1)
    for i in range(1, sink + 1):
        dfa.add_transition(g1.display(), i, i)

    dfa.add_transition(g2.display(), 1, 2)
    dfa.add_transition(g2.display(), 3, 4)
    dfa.add_transition(g2.display(), 0, 0)
    dfa.add_transition(g2.display(), 2, 2)

    dfa.add_transition(g3.display(), 1, 3)
    dfa.add_transition(g3.display(), 2, 4)
    dfa.add_transition(g3.display(), 0, 0)
    dfa.add_transition(g3.display(), 3, 3)

    dfa.add_transition(g23.display(), 1, 4)
    dfa.add_transition(g23.display(), 0, 0)
    dfa.add_transition(g23.display(), 2, 2)
    dfa.add_transition(g23.display(), 3, 3)

    dfa.toDot("DFA")
    dfa.prune_eff_transition()
    dfa.g_unsafe = 'obs'



    result = mdp.product(dfa, mdp)

    # result.AOpt = mdp.option_generation(dfa)

    # result.SVI(0.0000001)

    result.segmentation()
    # result.svi_plot_key = True


    # result.SVI(0.0000001)
    # result.
    # result.layer_plot()

    result.SVI_option(0.0000001)

    # result.layer_plot()
    # result.plot_curve(result.action_special, result.hybrid_special, "compare_V_3_3")
    # print result.hybrid_special






