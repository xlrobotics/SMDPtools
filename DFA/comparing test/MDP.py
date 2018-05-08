import numpy as np
import matplotlib.pyplot as plt
from pandas import *
from copy import deepcopy as dcp

import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly import tools
plotly.tools.set_credentials_file(username='SeanLau', api_key='d1fuOtEPoJU7V6GhntKN')

# tags:
# Init: used for initializing the object parameters
# Tool: inner function called by API, not transparent to the user
# API: can be called by the user
# VIS: visualization
# DISCARD: on developing or not related functions

class MDP():

    def __init__(self, alpha = 0.1, gamma = 0.9, epsilon = 0.2, num_actions = 4):
        self.alpha = alpha # not used
        self.gamma = gamma # discounting factor

        self.epsilon = epsilon # transition probability, set as constant here

        self.num_actions = num_actions # number of actions
        self.unsafe = {}

        self.ID = 'root' # MDP id to identify the layer of an MDP, set as 'root' if it's original MDP,
        # use option index as an ID if it is an option

        self.S = [] # state space e.g. [[0,0],[0,1]...]
        self.originS = []
        self.wall_cord = [] # wall state space, e.g. if state space is [[1,1]], wall is [[1,2], [2,1], [1,0], [0,1]]
        self.L = {} # labeling function
        self.Exp = {} # another way to store labeling pairs

        self.goal = [] # goal state set
        self.interruptions = [] # unsafe state set

        self.A = {"N":(-1, 0), "S":(1, 0), "W":(0, -1), "E":(0, 1)} # actions for grid world
        # what option does, should be a semi-Markov process, o = a1, a2, a3, a4, a5, a6..., aN

        self.V, self.V_ = {}, {} # value function and V_ denotes memory of the value function in last iteration
        self.init_V, self.init_V_ = {}, {} # store initial value function

        self.Q, self.Q_ = {}, {} # state action value function
        self.T = {} # terminal states
        self.Po = {} # transition matrix

        self.R = {} # reward function
        self.init_R = {} # initial reward function

        self.P = {} # 1 step transition probabilities
        # self.Product_MDP = MDP(alpha = 0.1, gamma = 0.9, epsilon = 0.2, num_actions = 4)

        self.Pi, self.Pi_ = {}, {} # policy and memory of last iteration policy
        self.Opt = {} #options
        self.AOpt = {} # atomic options without production

        self.dfa = None
        self.mdp = None
        self.plotKey = False
        self.svi_plot_key = False

        self.action_special = None
        self.hybrid_special = None

    # Tool: geometric series function, called by self.transition_matrix()
    def GeoSeries(self, P):
        sum = dcp(P)
        # method 1 series approximation with large number
        for k in range(2, 100):
            result = np.linalg.matrix_power(dcp(P), k)
            # for i in range(1,k):
            #     result *= P
            sum += result

        return sum

    # API: calculating transition matrix for an option
    def transition_matrix(self): # write the correct transition probability according to the policy
        PP = {}
        sink = "sink"
        PP[sink] = {}
        PP[sink][sink] = 1.0

        for s in self.S:
            s = tuple(s)
            PP[s] = {}

            PP[s][s] = 0.0
            PP[sink][s] = 0.0

            for s_ in self.S: # initialization
                s_ = tuple(s_)
                PP[s][s_] = 0.0

            if s not in self.goal:
                PP[s][sink] = 1 - self.gamma
            else:
                PP[s][sink] = 0.0
                PP[s][s] = 1.0

            if s in self.unsafe or s in self.interruptions:
                PP[s][sink] = 1.0

            if s not in self.goal and s not in self.unsafe and s not in self.interruptions:
                for a in self.Pi[s]:
                    for nb in self.P[s, a]:
                        PP[s][nb] = self.P[s, a][nb] * self.Pi[s][a]

        reference = DataFrame(dcp(PP)).T.fillna(0.0)

        PP = DataFrame(PP, reference.columns, reference.index)

        for key in PP.keys():
            for key_ in PP[key].keys():
                if PP[key][key_] < 1.0 and key_ != sink:
                    PP[key][key_] *= self.gamma #self.gamma

        sum = self.GeoSeries(dcp(PP))

        result = DataFrame(sum, PP.columns, PP.index)


        #normalize the result!!! Done
        for s1 in result.keys():
            try:
                sumResult = sum(result[s1].values())
                for s2 in result[s1].keys():
                    result[s1][s2] = result[s1][s2] / sumResult
            except:
                sumResult = result[s1].sum()
                for s2 in result[s1].keys():
                    result[s1][s2] = result[s1][s2] / sumResult

        self.TransitionMatrix = dcp(result)

    # Init: set labeling function
    def set_L(self, label):
        self.L = label

    # Init: set
    def set_Exp(self, exp):
        self.Exp = exp

    # Tool: a small math tool for the state space cross product
    def crossproduct(self, a, b):
        return [(tuple(y), x) for x in a for y in b]

    def option_composition(self, ctype, Opt_list):
        result = {}
        if ctype == 'disjunction':
            for state in self.originS:
                s = tuple(state)
                result[s] = sum([self.AOpt[name].V[s] for name in Opt_list])/len(Opt_list)

        if ctype == 'conjunction':
            for state in self.originS:
                s = tuple(state)
                result[s] = sum([self.AOpt[name].V[s] for name in Opt_list]) / len(Opt_list)

        self.Opt[Opt_list, ctype] = MDP()

    def option_generation(self, dfa):

        goals = dcp(self.Exp)
        del goals["phi"]

        delList = []
        for exp in goals:
            if exp not in dfa.effTS:
                delList.append(exp)

        unsafe_map = {}
        for delExp in delList:
            unsafe_map[delExp] = dcp(goals[delExp])
            del goals[delExp]

        filtered_S = []
        # print 'wall', self.wall_cord
        for s in self.S:  # new_S
            if s not in self.wall_cord:  # new_wall
                filtered_S.append(s)

        AOpt = {}
        # for qs in dfa.state_info.keys():
        opstacles = set([])
        for exp in goals:

            AOpt[exp] = MDP()
            AOpt[exp].ID = exp
            AOpt[exp].plotKey = True
            # AOpt[exp].unsafe = self.crossproduct(dfa.sink_states, filtered_S)

            AOpt[exp].set_S(filtered_S)
            AOpt[exp].goal = dcp(goals[exp])
            AOpt[exp].T = dcp(goals[exp])
            AOpt[exp].T += self.Exp[dfa.g_unsafe]

            AOpt[exp].P = dcp(self.P)

            for state in filtered_S:
                s = tuple(state)
                for a in self.A:
                    AOpt[exp].P[s, a] = {}
                    s_ = tuple(np.array(s) + np.array(self.A[a]))
                    if (s, a, s_) in self.P:
                        AOpt[exp].P[s, a][s_] = self.P[s, a, s_]

            for s in filtered_S:
                AOpt[exp].V[tuple(s)], AOpt[exp].V_[tuple(s)] = 0.0, 0.0
            for goal in goals[exp]:
                AOpt[exp].V[goal], AOpt[exp].V_[goal] = 100.0, 0.0

            AOpt[exp].R = {}
            for state in filtered_S:
                s = tuple(state)
                for a in self.A:
                    if (s, a) in AOpt[exp].P:
                        AOpt[exp].R[s, a] = 0

            AOpt[exp].SVI(0.000001)
            AOpt[exp].transition_matrix()

        return dcp(AOpt)
    # API: option segmentation from product state space based on DFA transitions: TODO: might be discarded later!!!
    def segmentation(self):
        goals = dcp(self.mdp.Exp)
        TS_tree = dcp(self.dfa.transition_tree)
        del goals["phi"]

        delList = []
        for exp in goals:
            if exp not in self.dfa.effTS:
                delList.append(exp)

        unsafe_map = {}
        for delExp in delList:
            unsafe_map[delExp] = dcp(goals[delExp])
            del goals[delExp]

        # print delList
        print unsafe_map

        print 'unsafe',self.unsafe
        print "!!!!!",self.T
        print self.P
        print self.dfa.sink_states
        print self.dfa.transition_tree

        # for element in self.P:
        #     if "sink" in self.P[element]:
        #         print element, "sink"

        q_unsafe = dcp(self.dfa.sink_states)
        # print self.mdp.Exp
        print 'dfa_transitions: ', self.dfa.state_transitions
        # collect transitions into efficient transition, get rid of same state transition and sink transition
        print 'effTs', self.dfa.effTS
        print 'goals: ', goals
        for exp in goals:
            # TODO: need to get the transition!!!
            # print exp, goals[exp]
            # print exp
            for transitions in self.dfa.effTS[exp]:
                print "transition", transitions
                self.Opt[exp, tuple(transitions)] = MDP()
                self.Opt[exp, tuple(transitions)].ID = exp, tuple(transitions)
                self.Opt[exp, tuple(transitions)].plotKey = True
                self.Opt[exp, tuple(transitions)].unsafe = dcp(self.unsafe)
                # self.Opt[exp, tuple(transitions)].A = dcp(self.A)

                # TODO special transition need to deal with, if the transition share the same origin as current
                # TODO but different non-fail target, need to get rid of it!!
                interruptions = {}
                for target in TS_tree[transitions[0]]:
                    if target != transitions[1] and target not in q_unsafe:
                        for ts in self.dfa.invEffTS[transitions[0], target]:
                            if ts not in interruptions:
                                interruptions[ts] = target # {:}
                print "interruptions: ", interruptions

                # interruptions = {} # ignore the interruptions

                S = []
                g = []
                for state in self.S:
                    # print state
                    if state[0] not in goals[exp]:
                        # print self.mdp.L[state[0]]
                        if self.mdp.L[state[0]] in unsafe_map:
                            # print transitions, state
                            if state[1] in q_unsafe and state not in S:
                                print "unsafe", state
                                S.append(state)
                        elif self.mdp.L[state[0]] in interruptions:
                            if state[1] == interruptions[self.mdp.L[state[0]]] and state not in S:
                                S.append(state)
                                self.Opt[exp, tuple(transitions)].interruptions.append(state)
                        else:
                            if state[1] == transitions[0] and state not in S:
                                S.append(state)
                    else:
                        if state[1] == transitions[1] and state not in S:
                            S.append(state)
                            if state not in g:
                                g.append(state)

                print "states", S
                print "goals", g


                self.Opt[exp, tuple(transitions)].set_S(S)

                self.Opt[exp, tuple(transitions)].goal = dcp(g)
                self.Opt[exp, tuple(transitions)].T = dcp(g)

                P = {}
                for prior in self.P.keys():
                    if prior[0] in S and prior[0] not in g:
                        if prior not in P:
                            P[prior] = dcp(self.P[prior])

                self.Opt[exp, tuple(transitions)].P = dcp(P)

                print "!!!!!!!!!!",len(self.Opt[exp, tuple(transitions)].P.keys())
                print len(self.Opt[exp, tuple(transitions)].S)
                print "probability", P

                # value function initiation, if doing this step, no need to assign value to reward function
                for s in S:
                    self.Opt[exp, tuple(transitions)].V[s] = 0
                    self.Opt[exp, tuple(transitions)].V_[s] = 0
                for goal in g:
                    self.Opt[exp, tuple(transitions)].V[goal] = 100.0
                    self.Opt[exp, tuple(transitions)].V_[goal] = 0

                self.Opt[exp, tuple(transitions)].R = dcp(self.R)

                self.Opt[exp, tuple(transitions)].SVI(0.000001)
                print "policy", self.Opt[exp, tuple(transitions)].Pi
                self.Opt[exp, tuple(transitions)].transition_matrix()
                # print "++++++++++"
                # for s in self.Opt[exp, tuple(transitions)].S:
                #     print self.Opt[exp, tuple(transitions)].TransitionMatrix[s][]

        return 0

    # API: LTL*MDP product
    def product(self, dfa, mdp):
        result = MDP()

        new_S = self.crossproduct(dfa.states, mdp.S)
        new_wall = self.crossproduct(dfa.states, mdp.wall_cord)

        new_A = mdp.A

        filtered_S = []
        for s in mdp.S: #new_S
            if s not in mdp.wall_cord: # new_wall
                filtered_S.append(s)

        result.originS = dcp(filtered_S)
        # print filtered_S

        new_sink = self.crossproduct(dfa.final_states, filtered_S)
        new_unsafe = self.crossproduct(dfa.sink_states, filtered_S)
        # print mdp.L
        new_P = {}
        new_R = {}

        new_Q = {}
        new_Q_ = {}

        new_V = {}
        new_V_ = {}

        true_new_s = []

        sink = "fail"

        for p in mdp.P.keys():

            for q in dfa.states:
                new_s = (p[0], q)
                new_a = p[1]
                if new_s not in new_sink: #(mdp.L[p[2]].display(), q) in dfa.state_transitions

                    q_ = dfa.state_transitions[mdp.L[p[2]], q]
                    new_s_ = (p[2], q_)

                    if (new_s, new_a) not in new_P:
                        new_P[new_s, new_a] = {}
                    new_P[new_s, new_a][new_s_] = mdp.P[p]

                    if tuple(new_s_) not in true_new_s:
                        true_new_s.append(tuple(new_s_))
                else:
                    new_s_ = sink

                    if (new_s, new_a) not in new_P:
                        new_P[new_s, new_a] = {}

                    new_P[new_s, new_a][new_s_] = 1

                if q in dfa.final_states and q not in dfa.sink_states:
                    new_R[new_s, new_a] = 0
                    new_V[new_s] = 100.0
                    new_V_[new_s] = 0
                elif q in dfa.sink_states:
                    new_R[new_s, new_a] = 0
                    new_V[new_s] = 0  # -1
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
        # print len(self.P), len(new_P)
        # print len(true_new_s), len(new_S) - len(new_wall)

        result.set_S(true_new_s)


        print "result S length: ",len(result.S)

        result.P = dcp(new_P)
        result.A = dcp(new_A)
        result.R = dcp(new_R)
        result.T = dcp(new_sink)
        result.V = dcp(new_V)
        result.V_ = dcp(new_V_)
        result.dfa = dcp(dfa)
        result.mdp = dcp(mdp)
        result.unsafe = dcp(new_unsafe)

        result.init_V, result.init_V_ = dcp(new_V), dcp(new_V_)
        result.init_R = dcp(new_R)

        # for g in result.T:
        #     print g, result.V[g]

        # self.SVI(true_new_s, new_A, new_P, new_R, new_sink, new_V, new_V_, 0.000001, dfa, mdp)
        # self.SVI_option(true_new_s, new_A, new_P, new_R, new_sink, new_V, new_V_, 0.000001, dfa, mdp)
        return result

    # Init: preparation for 1 step transition probability generation, not important
    def add_wall(self, inners):
        wall_cords = []
        for state in inners:
            for action in self.A:
                # print state, self.A[action]
                temp = list(np.array(state) + np.array(self.A[action]))
                if temp not in inners:
                    wall_cords.append(temp)
        return wall_cords

    # Init: preparation for 1 step transition probability generation, not important
    def set_WallCord(self, wall_cord):
        for element in wall_cord:
            self.wall_cord.append(element)
            if element not in self.S:
                self.S.append(element)

    # Init: init state space
    def set_S(self, in_s = None):
        if in_s == None:
            for i in range(13):
                for j in range(13):
                    self.S.append([i,j])
        else:
            self.S = dcp(in_s)

    # Init: init goal state set
    def set_goal(self, goal=None, in_g=None):
        if goal!=None and in_g == None:
            self.goal.append(goal)
        if goal == None and in_g!=None:
            self.goal = dcp(in_g)

    # Init: init state value function
    def init_V(self):
        for state in self.S:
            s = tuple(state)
            if s not in self.V:
                self.V[s], self.V_[s] = 0.0, 0.0
            if state in self.goal:
                self.V[s], self.V_[s] = 100.0, 0.0

    # Init: generating 1 step transition probabilities for 2d grid world
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

                        for k in range(1, 3):
                            # if s == (1, 1):
                            #     print a, AS[k], k, explore[k]
                            if a not in AS[k]:
                                for _s_ in explore[k]:

                                    self.P[s, a, tuple(_s_)] = self.epsilon/2.0
                                    # print "P("+str(tuple(_s_))+"|"+str(s)+","+str(a)+") =", self.P[s, a, tuple(_s_)]

    # Tool: turning dictionary structure to vector
    def Dict2Vec(self, V, S):
        # print S
        v = []
        for s in S:
            v.append(V[tuple(s)])
        return np.array(v)

    # Tool: sum operator in value iteration algorithm, called by self.SVI()
    def Sigma_(self, s, a):
        total = 0.0
        # print P[s, a]
        # if s[0] == (2, 2):
        #     print s, a, P[s, a].keys()

        if self.ID != "root":
            if s in self.unsafe:
                # print "unsafe", self.P[s, a]
                return total
            elif s in self.interruptions:
                # print "interruption", self.P[s, a]
                return total + 1.0

        for s_ in self.P[s, a].keys():
            # print self.V_[s_]
            total += self.P[s, a][s_] * self.V_[s_]
        return total

    # Tool: sum operator in value iteration algorithm, called by self.SVI_option()
    def Sigma_opt(self, s, opt):

        # print self.Opt[opt].S
        # print opt, self.Opt[opt].goal
        # print self.Opt[opt].wall_cord

        # g = tuple(self.Opt[opt].goal[0])
        total = 0
        # method 1: sigma P(s,o',s')*V(s')
        # total = 0.0
        # for state in self.Opt[opt].S:
        #     if state not in self.Opt[opt].wall_cord and state in self.Opt[opt].goal: #state in self.Opt[opt].goal
        #         s_ = tuple(state)
        #         # print s, s_
        #         total += self.Opt[opt].TransitionMatrix[s][s_] * self.V_[s_]
        #         print self.V_[s_], total
        for g in self.Opt[opt].goal:
            # print self.Opt[opt].TransitionMatrix[s][g]
            total += self.Opt[opt].TransitionMatrix[s][g] * self.V_[g]
        # print self.V_[g], total, self.Opt[opt].TransitionMatrix[s][g]
                # print "opt",total, self.Opt[opt].TransitionMatrix[s, s_]
        # print total

        # method 2: V_opt(s) * V(s,o)
        # g = tuple(self.Opt[opt].goal[0])
        # total = self.Opt[opt].V[s] * self.V_[g] # self.V[g]


        return total

    # API: option && action hybrid SVI runner
    def SVI_option(self, threshold):
        tau = 1.0
        self.V, self.V_ = dcp(self.init_V), dcp(self.init_V_)
        self.R = dcp(self.init_R)

        self.Pi, self.Pi_ = {}, {}

        V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)

        it = 1
        diff = []
        val = []
        pi_diff = []
        special = []
        diff.append(np.inner(V_current - V_last, V_current - V_last))

        while np.inner(V_current - V_last, V_current - V_last) > threshold:
            if it > 24:
                break
            # print self.V
            val.append(np.linalg.norm(V_current))

            # self.layer_plot('iteration '+str(it),'RL/LTL_SVI_opt/')
            # pi_diff.append(self.policy_diff(self.Pi, self.Pi_))
            special.append(self.V[(3, 3),0])

            for s in self.S:
                self.V_[s] = self.V[s]

                if s not in self.Pi: # for softmax
                    self.Pi[s] = {}
                if s not in self.Q:
                    self.Q[s] = {}

            for s in self.S:
                # print "state print:", s, tuple(s)
                if s not in self.T:
                    max_va, max_vo = -0.01, -0.01
                    max_a, max_opt = None, None

                    # for a in self.A:
                    #     if (tuple(s), a) in self.P:
                    #
                    #         v = self.R[tuple(s), a] + self.gamma * self.Sigma_(tuple(s), a)
                    #         self.Q[tuple(s)][a] = np.exp(v / tau) # softmax solution
                    #
                    #         if v > max_va:
                    #             max_va, max_a = v, a

                    for opt in self.Opt.keys():
                        if s in self.Opt[opt].S and s not in self.Opt[opt].goal and s not in self.Opt[opt].unsafe:
                            # print self.Opt[opt].transition_matrix
                            if tuple([tuple(s), opt]) not in self.R:
                                self.R[tuple(s), opt] = 0.0

                            v = self.R[tuple(s), opt] + self.Sigma_opt(tuple(s), opt)
                            self.Q[tuple(s)][opt] = np.exp(v / tau) # softmax solution
                                # print "optv", v
                            if v > max_vo:
                                max_vo, max_opt = v, opt

                    # if max_va > max_vo:
                    #     self.V[tuple(s)], self.Pi[tuple(s)] = max_va, max_a
                    # else:
                    #     self.V[tuple(s)], self.Pi[tuple(s)] = max_vo, max_opt

                    # softmax solution
                    sumQ = sum(self.Q[tuple(s)].values())
                    if sumQ < 1:
                        print "sumQ",sumQ, s
                    for choice in self.Q[tuple(s)]:
                        self.Pi[tuple(s)][choice] = self.Q[tuple(s)][choice]/sumQ
                    self.V[tuple(s)] = tau * np.log(sumQ)
                else:
                    if s not in self.unsafe:
                        self.V[tuple(s)], self.Pi[tuple(s)] = 100.0, None
                    else:
                        self.V[tuple(s)], self.Pi[tuple(s)] = 0.0, None

            V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
            diff.append(np.inner(V_current - V_last, V_current - V_last))
            # print V_current, V_last
            it += 1
        print "difference: ", diff
        print "iterations: ", it
        self.hybrid_diff = diff
        self.hybrid_val = val
        # self.hybrid_pi_diff = pi_diff
        self.hybrid_special = special
        return 0

    # API: action SVI runner
    def SVI(self, threshold):
        # S, A, P, R, sink, V, V_ = inS, inA, inP, inR, in_sink, inV, inV_
        tau = 1.0

        for g in self.T:
            print "init_V", g, self.V[g]

        self.Pi, self.Pi_ = {}, {}

        Difference_record = []
        V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
        # Difference_record.append()
        # print self.V
        it = 1
        diff = []
        val = []
        pi_diff = []
        special = []
        # print self.wall_cord
        # print V_current, V_last, np.inner(V_current - V_last, V_current - V_last)
        diff.append(np.inner(V_current - V_last, V_current - V_last))
        # if self.plotKey:
        #     print "model info: +++++++++"
        #     print self.T
        #     print self.A
        #     print self.P
        #     print "model info: ---------"

        while np.inner(V_current - V_last, V_current - V_last) > threshold: # np.inner(V_current - V_last, V_current - V_last) > threshold
            # diff = np.inner(V_current - V_last, V_current - V_last)
            # print diff
            # Difference_record.append(diff)
            # if self.svi_plot_key:
            #     self.layer_plot('iteration ' + str(it), 'RL/LTL_SVI_act/')
            # print 'difference:', np.inner(V_current - V_last, V_current - V_last)
            val.append(np.linalg.norm(V_current))

            if tuple([(3, 3), 0]) in self.V:
                special.append(self.V[(3, 3), 0])

            for s in self.S:
                self.V_[tuple(s)] = self.V[tuple(s)]

                if tuple(s) not in self.Pi: # for softmax
                    self.Pi[tuple(s)] = {}
                if tuple(s) not in self.Q:
                    self.Q[tuple(s)] = {}
                # if tuple(s) in self.Pi:
                #     self.Pi_[tuple(s)] = self.Pi[tuple(s)]

            for s in self.S:

                if tuple(s) not in self.T:
                    # print s, self.T
                    max_v, max_a = -0.001, None

                    for a in self.A:
                        if (tuple(s), a) in self.P:

                            v = self.R[tuple(s), a] + self.gamma * self.Sigma_(tuple(s), a)

                            self.Q[tuple(s)][a] = np.exp(v / tau)  # softmax solution
                            # if self.plotKey:
                            #     print v, self.R[tuple(s), a], self.Sigma_(tuple(s), a)

                            if v > max_v:
                                max_v, max_a = v, a

                    # self.V[tuple(s)], self.Pi[tuple(s)] = max_v, max_a

                    # softmax solution
                    # print self.Q[tuple(s)]
                    sumQ = sum(self.Q[tuple(s)].values())
                    for choice in self.Q[tuple(s)]:
                        self.Pi[tuple(s)][choice] = self.Q[tuple(s)][choice] / sumQ
                    # self.V[tuple(s)] = tau * np.log(sumQ)
                    self.V[tuple(s)] = tau * np.log(sumQ)

                # else

            V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
            diff.append(np.inner(V_current - V_last, V_current - V_last))
            # if self.plotKey:
            #     print self.gamma
            #     print self.V
            #     print self.P
            # print V_current, V_last
            it += 1

        print 'iterations taken:', it
        # print Policy

        # policy testing with different initial states
        # for s in mdp.S:
        #     if s not in mdp.wall_cord:
        #         start_point = tuple([tuple(s),0])
        #         print self.testPolicy(start_point, Policy, S, A, V, P, dfa, mdp)

        # self.plot("result")
        for g in self.T:
            print g, self.V[g]

        if self.plotKey:
            self.plot_heat(self.ID)

        if not self.plotKey:
            print "difference: ", val

        self.action_diff = diff
        self.action_val = val
        # self.action_pi_diff = pi_diff
        self.action_special = special

        return 0

    # DISCARD
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
            # print "through policy action:", act, "->", cs, V[cs], "(objective value)", cp, "chain probability"
            # if nq == 3:
            #     return "task succeed"
            # if nq == 4:
            #     return "task failed"

    # VIS: plotting heatmap using either matplotlib ot plotly to visualize value function for all options
    def layer_plot(self, title, dir=None):
        z = {}
        for key in self.Opt:
            # print self.Opt[key].S,  len(self.Opt[key].S)
            q1 = key[1][0]
            q2 = key[1][1]
            g = key[0]

            # print self.V
            temp = np.random.random((6, 8))
            for state in self.Opt[key].S:
                    # key = ((i, j), )
                    # if (i, j) not in self.V:
                    #     temp[(i, j)] = -1
                # print state[0], state
                i, j = 5-(state[0][0]-1), state[0][1]-1
                temp[i, j] = self.V[state]

            name = "layer-" + str(q1) + "-" + str(g) + "-" + str(q2)
            z[name] = temp

            # plt.figure()
            # plt.imshow(temp, cmap='hot', interpolation='nearest')

            # folder = "" if dir == None else dir

            # plt.savefig(folder + name + ".png")  # "../DFA/comparing test/"

        # title = "test_layers"
        # fname = 'RL/LTL_SVI/' + title
        fname = dir + title

        trace = []
        names = z.keys()
        fig = tools.make_subplots(rows=2, cols=2,
                                  subplot_titles=(names[0], names[1], names[2], names[3])
                                  )
        # specs = [[{'is_3d': True}, {'is_3d': True}], [{'is_3d': True}, {'is_3d': True}]]
        # subplot_titles=(names[0], names[1], names[2], names[3])

        trace.append(go.Heatmap(z=z[names[0]], showscale=True, colorscale='Jet'))
        trace.append(go.Heatmap(z=z[names[1]], showscale=False, colorscale='Jet'))
        trace.append(go.Heatmap(z=z[names[2]], showscale=False, colorscale='Jet'))
        trace.append(go.Heatmap(z=z[names[3]], showscale=False, colorscale='Jet'))

        fig.append_trace(trace[0], 1, 1)
        fig.append_trace(trace[1], 1, 2)
        fig.append_trace(trace[2], 2, 1)
        fig.append_trace(trace[3], 2, 2)

        # py.iplot(
        #     [
        #      dict(z=z[0]+1.0, showscale=False, opacity=0.9, type='surface'),
        #      dict(z=z[1]+2.0, showscale=False, opacity=0.9, type='surface'),
        #      dict(z=z[2]+3.0, showscale=False, opacity=0.9, type='surface'),
        #      dict(z=z[3]+4.0, showscale=False, opacity=0.9, type='surface')],
        #     filename=fname)

        fig['layout'].update(title=title)

        py.iplot(fig, filename=fname)

        return 0

    # VIS: plotting single heatmap when necessary, used for debugging!!
    def plot_heat(self, key, dir=None):

        g, q1, q2 = None, None, None
        folder = "" if dir == None else dir



        if type(key) == type("string"):
            name = "reaching option" + key
            plt.figure()

            temp = np.random.random((6, 8))
            for state in self.S:
                if state not in self.wall_cord:
                    temp[state[0] - 1, state[1] - 1] = self.V[tuple(state)]

            plt.imshow(temp, cmap='hot', interpolation='nearest')
            plt.savefig(folder + name + ".png")
            return 0
        else:

            temp = np.random.random((6, 8))
            for state in self.S:
                temp[state[0][0] - 1, state[0][1] - 1] = self.V[state]

            q1 = key[1][0]
            q2 = key[1][1]
            g = key[0]


        name = "layer-" + str(q1) + "-" + str(g) + "-" + str(q2)
        plt.figure()
        plt.imshow(temp, cmap='hot', interpolation='nearest')
        plt.savefig(folder + name + ".png")  # "../DFA/comparing test/"

        ct = go.Contour(
            z=temp,
            type='surface',
            colorscale='Jet',
        )
        fname = 'RL/LTL_SVI/' + name

        py.iplot(
            [   ct,
                dict(z=temp, showscale=False, opacity=0.9, type='surface')],
            filename=fname)

        return 0

    # VIS: comparing the trend of two curves for any function or variable
    def plot_curve(self, trace1, trace2, name):
        plt.figure()

        # print x
        # print trace1
        # print trace2
        l1, = plt.plot(trace1, label="action")
        l2, = plt.plot(trace2, label="hybrid")

        plt.legend(handles=[l1, l2])

        plt.ylabel('Value iteration difference')
        plt.savefig(name + ".png")


