import numpy as np
import matplotlib.pyplot as plt
from pandas import *

# a method to find optimal or epsilon-optimal policy
#
# simplified, R(s, a, s') -> R(s, a), and since in grid world the next state is deterministic given (s, a)
# P(j|s,a) = 1 for some j
#
# 1:           Procedure Value_Iteration(S,A,P,R,theta)
# 2:           Inputs
# 3:                     S is the set of all states
# 4:                     A is the set of all actions
# 5:                     P is state transition function specifying P(s'|s,a)
# 6:                     R is a reward function R(s,a,s')
# 7:                     theta a threshold, theta > 0
# 8:           Output
# 9:                     pi[S] approximately optimal policy
# 10:                    V[S] value function
# 11:          Local
# 12:                    real array Vk[S] is a sequence of value functions
# 13:                    action array pi[S]
#
# 14:          assign V0[S] arbitrarily
# 15:          k <- 0
# 16:          repeat
# 17:                    k <- k+1
# 18:                    for each state s do
# 19:                       Vk[s] = max_a( R(s, a) + lambda * Vk-1[j] )
# 20:           until for all s satisfies |Vk[s]-Vk-1[s]| < theta
# 21:           for each state s do
# 22:                     pi[s] = argmax_a( R(s, a) + lambda * sum_j( P(j|s,a) * Vk[j]) ) )
# 23:           return pi,Vk

# initial value 0, reward +1, discount gamma = 0.9, pi_greedy = 2/3, pi_others = ( 1/3 ) / 3

# Difference to q-learning, q-learning doesn't

# For options, P(s, o, s') = sum_k ( gamma^k * P(arrive in k times) ), given learned policy, say transition mat P1 \
# P = sum_k ( gamma^k * P1^k ) = (1 - P1)^(-1) according to...


class MDP():

    def __init__(self, alpha = 0.1, gamma = 0.9, epsilon = 0.2, num_actions = 4):
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon

        self.num_actions = num_actions

        self.S = []
        self.L = {}

        self.goal = []

        self.A = {"N":(-1, 0), "S":(1, 0), "W":(0, -1), "E":(0, 1)}
        # what option does, should be a semi-Markov process, o = a1, a2, a3, a4, a5, a6..., aN

        self.V = {}
        self.V_ = {}
        self.T = {}

        self.R = {}
        self.P = {}

        self.Policy = {}

        self.wall_cord = []

    def set_L(self, label):
        self.L = label

    def crossproduct(self, a, b):
        return [(tuple(y), x) for x in a for y in b]

    def product(self, dfa, mdp):
        result = MDP()
        new_S = self.crossproduct(dfa.states, mdp.S)
        new_wall = self.crossproduct(dfa.states, mdp.wall_cord)

        new_A = mdp.A

        filtered_S = []
        for s in mdp.S: #new_S
            if s not in mdp.wall_cord: # new_wall
                filtered_S.append(s)
        print filtered_S

        new_sink = self.crossproduct(dfa.final_states, filtered_S)
        # print mdp.L
        new_P = {}
        new_R = {}

        new_V = {}
        new_V_ = {}

        true_new_s = []
        print new_sink

        sink = "sink"

        for p in mdp.P.keys():
            # if [(p[0], i) in new_S for i in dfa.states]:
            #     print p[0], p[1], p[2], mdp.L[p[2]].display(), "--"
            #
            # else:
            #     print p[0], p[1], p[2], mdp.L[p[2]].display(), ".."

            for q in dfa.states:
                new_s = (p[0], q)
                new_a = p[1]
                if new_s not in new_sink: #(mdp.L[p[2]].display(), q) in dfa.state_transitions

                    q_ = dfa.state_transitions[mdp.L[p[2]].display(), q]
                    new_s_ = (p[2], q_)

                    if (new_s, new_a) not in new_P:
                        new_P[new_s, new_a] = {}
                    new_P[new_s, new_a][new_s_] = mdp.P[p]

                    if list(new_s_) not in true_new_s:
                        true_new_s.append(list(new_s_))
                else:
                    new_s_ = sink

                    if (new_s, new_a) not in new_P:
                        new_P[new_s, new_a] = {}

                    new_P[new_s, new_a][new_s_] = 1

                if q in dfa.final_states and q not in dfa.sink_states:
                    new_R[new_s, new_a] = 0
                    new_V[new_s] = 1
                    new_V_[new_s] = 0
                elif q in dfa.sink_states:
                    new_R[new_s, new_a] = 0
                    new_V[new_s] = -1
                    new_V_[new_s] = 0
                elif q not in dfa.final_states:
                    new_R[new_s, new_a] = 0
                    new_V[new_s] = 0
                    new_V_[new_s] = 0

                if new_s not in true_new_s:
                    true_new_s.append(tuple(new_s))

        #
        # for key in new_P.keys():
        #     print key, new_P[key]
        print len(self.P), len(new_P)
        print len(true_new_s), len(new_S) - len(new_wall)

        self.SVI(true_new_s, new_A, new_P, new_R, new_sink, new_V, new_V_, 0.000001, dfa, mdp)




    def add_wall(self, inners):
        wall_cords = []
        for state in inners:
            for action in self.A:
                # print state, self.A[action]
                temp = list(np.array(state) + np.array(self.A[action]))
                if temp not in inners:
                    wall_cords.append(temp)
        return wall_cords


    def set_WallCord(self, wall_cord):
        for element in wall_cord:
            self.wall_cord.append(element)
            if element not in self.S:
                self.S.append(element)

    def set_S(self, in_s = None):
        if in_s == None:
            for i in range(13):
                for j in range(13):
                    self.S.append([i,j])
        else:
            self.S = in_s

    def set_goal(self, goal):
        self.goal.append(goal)

    def init_V(self):
        for state in self.S:
            s = tuple(state)
            if s not in self.V:
                self.V[s], self.V_[s] = 0.0, 0.0
            if state in self.goal:
                self.V[s], self.V_[s] = 1.0, 0.0

    # def set_R(self):
        # set all rewards to 0

    # def set_P(self, s, a, s_, p):
    #     self.P[s, a, s_] = p

    def set_P(self):
        for state in self.S:
            if state not in self.wall_cord:
                s = tuple(state)
                explore = {1:[], 2:[]}
                AS = {1:["N", "S"], 2:["E", "W"]}

                for i in range(1, 3):
                    for act in AS[i]:
                        temp = tuple(np.array(s) + np.array(self.A[act]))
                        if list(temp) not in self.wall_cord:
                            explore[i].append(temp)
                        else:
                            explore[i].append(s)


                for a in self.A.keys():
                    s_ = tuple(np.array(s) + np.array(self.A[a]))

                    if list(s_) not in self.wall_cord:

                        self.P[s, a, s_] = 1 - self.epsilon
                        print "P("+str(s_)+"|"+str(s)+","+str(a)+") =", self.P[s, a, s_]
                        # if s == (1,2):
                        #     string = s, a, s_
                        #     print string, self.P[string]

                        for k in range(1, 3):
                            # if s == (1, 1):
                            #     print a, AS[k], k, explore[k]
                            if a not in AS[k]:
                                for _s_ in explore[k]:

                                    self.P[s, a, tuple(_s_)] = self.epsilon/2.0
                                    print "P("+str(tuple(_s_))+"|"+str(s)+","+str(a)+") =", self.P[s, a, tuple(_s_)]



    def Dict2Vec(self, V, S):
        v = []
        for s in S:
            v.append(V[tuple(s)])
        return np.array(v)

    def Sigma(self, s, a, A, P, V_):
        total = 0.0
        # print P[s, a]
        for s_ in P[s, a].keys():
            total += P[s, a][s_] * V_[s_]
        return total


    def SVI(self, inS, inA, inP, inR, in_sink, inV, inV_, threshold, dfa, mdp):
        S, A, P, R, sink, V, V_ = inS, inA, inP, inR, in_sink, inV, inV_
        Policy = {}

        V_current, V_last = self.Dict2Vec(V, S), self.Dict2Vec(V_, S)

        it = 1
        # print self.wall_cord
        # print V_current, V_last, np.inner(V_current - V_last, V_current - V_last)

        while np.inner(V_current - V_last, V_current - V_last) > threshold: # np.inner(V_current - V_last, V_current - V_last) > threshold

            for s in S:
                V_[tuple(s)] = V[tuple(s)]

            for s in S:

                if tuple(s) not in sink:
                    max_v, max_a = -1.0 * 99999999, None

                    for a in A:
                        if (tuple(s), a) in P:
                            v = R[tuple(s), a] + self.gamma * self.Sigma(tuple(s), a, A, P, V_)
                            if v > max_v:
                                max_v, max_a = v, a

                    V[tuple(s)], Policy[tuple(s)] = max_v, max_a

            V_current, V_last = self.Dict2Vec(V, S), self.Dict2Vec(V_, S)
            # print V_current, V_last
            it += 1

        print it
        # print Policy

        # policy testing with different initial states
        for s in mdp.S:
            if s not in mdp.wall_cord:
                start_point = tuple([tuple(s),0])
                print self.testPolicy(start_point, Policy, S, A, V, P, dfa, mdp)
        # self.plot("result")
        return 0

    def testPolicy(self, ss, Policy, S, A, V, P, dfa, mdp):
        cs = ss
        cp = 1
        print "-----------------------------------------"
        print "start at:", cs, V[cs], "(objective value)"
        while True:
            act = Policy[cs]
            ns = tuple(np.array(cs[0]) + np.array(A[act]))
            nq = dfa.state_transitions[mdp.L[ns].display(), cs[1]]
            cp *= P[cs, act][tuple([ns, nq])]

            cs = tuple([ns, nq])
            print "through policy action:", act, "->", cs, V[cs], "(objective value)", cp, "chain probability"
            if nq == 3:
                return "task succeed"
            if nq == 4:
                return "task failed"


