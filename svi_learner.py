import numpy as np
import matplotlib.pyplot as plt
from pandas import *
import scipy
import copy

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


class GridRL():

    def __init__(self, alpha = 0.1, gamma = 0.9, lmbda=0.1, epsilon = 0.1, n = 13 * 13, num_actions = 4, room = 4):
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon # choice between deterministic and non-deterministic
        self.n = n
        self.num_actions = num_actions
        self.room = room

        self.S = []
        self.wall_cord = []
        self.Paths = []
        self.goal = []


        self.A = {"up":(0, 1), "down":(0, -1), "left":(-1, 0), "right":(1, 0)}
        self.NB = {} # knowledge base, used to store all logic, TODO: need to be modified
        self.RoomPath = {}

        # self.Opt = {("R1", "hw1"): None, ("R1", "hw2"): None, ("R3","hw3":None, "hw4":None}
        self.Opt = {}
        self.Beta = {} #termination condition
        # Beta only used for model free (e.g. Q-learning)

        # what option does, should be a semi-Markov process, o = a1, a2, a3, a4, a5, a6..., aN

        self.V, self.V_ = {}, {}
        self.Policy, self.Policy_ = {}, {}
        # self.Pi, self.Pi_ = {}, {}
        self.T = {}
        self.R = {}
        self.P = {}

        self.TransitionMatrix = None

        self.action_diff, self.hybrid_diff = [], []
        self.action_val, self.hybrid_val = [], []
        self.action_pi_diff, self.hybrid_pi_diff = [], []

        self.Ifplot = False

    def add_Opt(self, name, obj):
        if name not in self.Opt:
            self.Opt[name] = obj
            for state in self.S:
                self.R[tuple(state), name] = 0

    def get_RoomPath(self):
        return self.RoomPath

    def GeoSeries(self, P):
        sum = copy.deepcopy(P)

        for k in range(2, 50):
            result = np.linalg.matrix_power(copy.deepcopy(P), k)

            # for i in range(1,k):
            #     result *= P

            sum += result

        return sum




    def transition_matrix(self): # write the correct transition probability according to the policy
        PP = {}
        sink = "sink"
        # print self.Policy
        PP[sink] = {}
        PP[sink][sink] = 1.0

        mark = None

        for state in self.S:
            s = tuple(state)

            if state not in self.wall_cord:

                PP[s] = {}
                PP[s][s] = 0.0
                PP[sink][s] = 0.0

                for state_ in self.S: # initialization
                    if state_ not in self.wall_cord:
                        s_ = tuple(state_)
                        PP[s][s_] = 0.0

                if state not in self.goal:
                    PP[s][sink] = 1 - self.gamma
                else:
                    PP[s][sink] = 0.0

                for a in self.A.keys():
                    nb = tuple(np.array(s) + np.array(self.A[a]))
                    if state in self.goal:
                        PP[s][s] = 1.0
                    else:
                        if a == self.Policy[s]:
                            if nb == tuple(self.goal[0]):
                                mark = s
                                print "!!!"
                            PP[s][nb] = 2.0/3
                        else:
                            if list(nb) not in self.wall_cord:
                                PP[s][nb] = 1.0/9
                            if list(nb) in self.wall_cord:
                                PP[s][s] += 1.0/9

        reference = DataFrame(copy.deepcopy(PP)).T.fillna(0.0)

        PP = DataFrame(PP, reference.columns, reference.index)


        for key in PP.keys():
            for key_ in PP[key].keys():
                # only s,s' or s,goal
                if PP[key][key_] < 1.0 and key_ != sink:
                    PP[key][key_] *= self.gamma #self.gamma

        # check = []
        # g = tuple(self.goal[0])
        # for key in PP.keys():
        #     check.append(PP[key][g])
        # print check
        # print PP[mark][g]

        # for s1 in PP.keys():
        #     sum = 0.0
        #     for s2 in PP[s1].keys():
        #         sum += PP[s1][s2]
        #     print sum
        # print PP.columns
        # print PP.index
        # print PP
        # PP = DataFrame(PP).T.fillna(0.0)
        # print PP

        sum = self.GeoSeries(copy.deepcopy(PP))

        # QQ = PP*self.gamma

        # print PP[tuple(g)][tuple(g)]
        # print np.linalg.eigvals(PP)
        # print len(PP)

        # identity_minus_P = DataFrame(np.identity(len(PP)), PP.columns, PP.index) - copy.deepcopy(PP) # self.gamma * PP
        # # print np.linalg.det(identity_minus_P)
        # inv = DataFrame(np.linalg.inv(identity_minus_P.values), PP.columns, PP.index) - DataFrame(np.identity(len(PP)), PP.columns, PP.index)

        # print scipy.sparse.linalg.pinv(identity_minus_P.values)*identity_minus_P.values

        # print "======================================"
        # print np.identity(len(PP))
        # print identity_minus_P[(3, 6)][(3, 6)]
        # print self.gamma*PP
        # print np.identity(len(PP))
        # print identity_minus_P
        result = DataFrame(sum, PP.columns, PP.index)

        ###TODO normalize the result!!!
        for s1 in result.keys():
            sum = 0.0
            for s2 in result[s1].keys():
                sum += result[s1][s2]
            # print sum
            for s2 in result[s1].keys():
                result[s1][s2] = result[s1][s2]/sum


        # print type(identity_minus_P)
        # print g
        '''
        g = tuple(self.goal[0])
        print result[g][g]
        temp = {}
        pairs = {}
        for s in self.S:
            if s not in self.wall_cord:
                temp[result[tuple(s)][g]] = None
                temp[result[g][tuple(s)]] = None
                temp[result[tuple(s)][tuple(s)]] = None

        for s in self.S:
            for s_ in self.S:
                if s not in self.wall_cord and s_ not in self.wall_cord:
                    if result[tuple(s)][tuple(s_)] > 1.0:
                        pairs[tuple(s), tuple(s_), result[tuple(s)][tuple(s_)]] = None
        print pairs.keys()
        # print temp.keys()
        '''
        self.TransitionMatrix = copy.deepcopy(result)



    def landscape(self, s): #TODO robot should be able to learn itself about the different landscape, add simple CNN later
        # how to identify a doorway? from behavior and vision?
        s_ = {}
        s_["up"] = list(np.array(s) + np.array(self.A["up"]))
        s_["down"] = list(np.array(s) + np.array(self.A["down"]))
        s_["right"] = list(np.array(s) + np.array(self.A["right"]))
        s_["left"] = list(np.array(s) + np.array(self.A["left"]))

        w = self.wall_cord
        # print w

        if list(s) in w:
            # print s
            return 0, "w"

        if s_["up"] in w and s_["down"] in w and s_["left"] not in w and s_["right"] not in w: #horizon hallway
            return 1, "h"
        elif s_["left"] in w and s_["right"] in w and s_["up"] not in w and s_["down"] not in w: #vertical hallway
            return 2, "v"
        elif s_["left"] in w and s_["right"] not in w and s_["up"] in w and s_["down"] not in w: #left top corner
            return 3, "lt"
        elif s_["left"] in w and s_["right"] not in w and s_["up"] not in w and s_["down"] in w: #left down corner
            return 4, "rt"
        elif s_["left"] not in w and s_["right"] in w and s_["up"] in w and s_["down"] not in w: #right top corner
            return 5, "ld"
        elif s_["left"] not in w and s_["right"] in w and s_["up"] not in w and s_["down"] in w: #right down corner
            return 6, "rd"


        return 10, "n"

    def add_wall(self, inners):
        wall_cords = []
        for state in inners:
            for action in self.A:
                # print state, self.A[action]
                temp = list(np.array(state) + np.array(self.A[action]))
                if temp not in inners:
                    wall_cords.append(temp)
        return wall_cords

    def NB_initiation(self):

        self.NB[1] = {"in":[], "hw":[]}
        self.NB[2] = {"in":[], "hw":[]}
        self.NB[3] = {"in":[], "hw":[]}
        self.NB[4] = {"in":[], "hw":[]}

        cor = {"lt":[], "rt":[], "ld":[], "rd":[]}
        frame = {1:{}, 2:{}, 3:{}, 4:{}}
        hw = {"h":[], "v":[]}

        for state in self.S:

            s = tuple(state)

            # self.V[s] = 0 # comment out later!!!!!!

            value, tag = self.landscape(s)
            # self.V[s] = value
            if value in [1, 2]:
                hw[tag].append(s)
            elif value in [3, 4, 5, 6]:
                cor[tag].append(s)

        # print cor["rt"]
        # print cor["ld"]

        for i in range(1, 5):
            # print
            frame[i]["range1"] = range(cor["rt"][i - 1][0], cor["ld"][i - 1][0] + 1)
            frame[i]["range2"] = range(cor["rt"][i - 1][1], cor["ld"][i - 1][1] + 1)

        for state in self.S:
            s = tuple(state)
            if state not in self.wall_cord:
                for i in range(1, 5):
                    if s[0] in frame[i]["range1"] and s[1] in frame[i]["range2"]:
                        self.NB[i]["in"].append(state)
                        # self.V[s] = i
        # self.V[(7,9)] = 10

        #
        # set options by hand, should be autonomous in future settings
        self.NB[1]["options"] = {(3, 6): 2, (6, 2): 3}
        self.NB[2]["options"] = {(3, 6): 1, (7, 9): 4}
        self.NB[3]["options"] = {(6, 2): 1, (10, 6): 4}
        self.NB[4]["options"] = {(7, 9): 2, (10, 6): 3}
        for i in self.NB.keys():
            for key in self.NB[i]["options"].keys():
                indices = ("R"+str(i), "R"+str(self.NB[i]["options"][key]))
                self.RoomPath[indices] = key

        #

        tiny_SVIs = {}
        for i in range(1, len(self.NB) + 1):
            ID = "R" + str(i)
            tiny_SVIs[ID] = {}
            tiny_SVIs[ID]["inner"] = self.NB[i]["in"]
            tiny_SVIs[ID]["options"] = self.NB[i]["options"]
            tiny_SVIs[ID]["goal"] = self.NB[i]["options"]

            for k in tiny_SVIs[ID]["options"].keys():
                tiny_SVIs[ID]["inner"].append(list(k))

            tiny_SVIs[ID]["wall_cords"] = self.add_wall(tiny_SVIs[ID]["inner"])

        '''
        ## for testing
        for s in tiny_SVIs["R1"]['inner']:
            self.V[s] = 1

        for o in tiny_SVIs["R1"]["options"]:
            self.V[o] = 2

        for w in tiny_SVIs["R1"]["wall_cords"]:
            self.V[w] = 3


        print tiny_SVIs, len(self.NB)

        print self.V
        print hw

        self.plot("test_inners") # for testing
        '''

        return tiny_SVIs

    def set_WallCord(self, wall_cord):
        for element in wall_cord:
            self.wall_cord.append(element)

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
        # print self.S
        # print self.goal
        for state in self.S:
            s = tuple(state)
            if state not in self.goal:
                self.V[s], self.V_[s] = 0.0, 0.0
            if state in self.goal:
                self.V[s], self.V_[s] = 1.0, 0.0

            # print self.V[s], self.V_[s], s

        # print self.V
        # print self.V_
    def init_Policy(self):
        self.Policy, self.Policy_ = {}, {}
    # def set_R(self):
        # set all rewards to 0

    def set_P(self):
        # print self.S
        # print self.wall_cord
        for state in self.S:
            if state not in self.wall_cord:
                # print s
                s = tuple(state)
                neighbour = {}
                for a in self.A.keys():
                    nb = tuple(np.array(s) + np.array(self.A[a]))

                    if list(nb) in self.S: #!!!!TODO a typo here
                        neighbour[nb] = a

                # print neighbour

                for s_ in neighbour:

                    self.P[s, neighbour[s_], s_], self.R[s, neighbour[s_]] = 2.0/3, 0.0

                    for _s_ in neighbour:
                        if not _s_ == s_:
                            self.P[s, neighbour[s_], _s_] = (1 - self.P[s, neighbour[s_], s_])*1.0/len(neighbour)

    def Dict2Vec(self, V):
        v = []
        for s in self.S:
            v.append(V[tuple(s)])
        return np.array(v)

    def Sigma(self, s, a):
        total = 0.0

        neighbour = {}
        for act in self.A.keys():
            nb = tuple(np.array(s) + np.array(self.A[act]))
            if nb not in self.S:
                neighbour[nb] = act

        for s_ in neighbour:
            # print s_
            total += self.P[s, a, s_] * self.V_[s_]
        return total

    def Sigma_opt(self, s, opt):

        # print self.Opt[opt].S
        # print opt, self.Opt[opt].goal
        # print self.Opt[opt].wall_cord

        g = tuple(self.Opt[opt].goal[0])

        # method 1: sigma P(s,o',s')*V(s')
        # total = 0.0
        # for state in self.Opt[opt].S:
        #     if state not in self.Opt[opt].wall_cord and state in self.Opt[opt].goal: #state in self.Opt[opt].goal
        #         s_ = tuple(state)
        #         # print s, s_
        #         total += self.Opt[opt].TransitionMatrix[s][s_] * self.V_[s_]
        #         print self.V_[s_], total

        total = self.Opt[opt].TransitionMatrix[s][g] * self.V_[g]
        # print self.V_[g], total, self.Opt[opt].TransitionMatrix[s][g]
                # print "opt",total, self.Opt[opt].TransitionMatrix[s, s_]
        # print total

        # method 2: V_opt(s) * V(s,o)
        # g = tuple(self.Opt[opt].goal[0])
        # total = self.Opt[opt].V[s] * self.V_[g] # self.V[g]


        # print total, self.V_[g], opt

        return total

        # if total > 0 and self.V_[g] > 0:
        #     return total * self.V_[g]
        # else:
        #     return total
    def policy_diff(self, Pi_current, Pi_last):
        count = 0
        for key in Pi_current.keys():
            if key in Pi_current and key not in Pi_last:
                count += 1
            elif key in Pi_current and key in Pi_last:
                if Pi_current[key] != Pi_last[key]:
                    count += 1

        return count

    def SVI_action(self, threshold):

        V_current, V_last = self.Dict2Vec(self.V), self.Dict2Vec(self.V_)

        it = 1
        diff = []
        val = []
        pi_diff = []
        # print self.wall_cord
        # print V_current, V_last, np.inner(V_current - V_last, V_current - V_last)

        while np.inner(V_current - V_last, V_current - V_last) > threshold:
            if it > 24:
                break
            # print it
            val.append(np.linalg.norm(V_current))
            diff.append(np.inner(V_current - V_last, V_current - V_last))
            pi_diff.append(self.policy_diff(self.Policy, self.Policy_))

            if self.Ifplot == True:
                self.plot("Action_result" + str(it), "../SMDPtools/ActionHeat/")

            for state in self.S:
                s = tuple(state)
                if s in self.Policy:
                    self.Policy_[s] = self.Policy[s]
                self.V_[s] = self.V[s]

            for s in self.S:

                if s not in self.wall_cord and s not in self.goal:
                    max_v, max_a = -1.0 * 99999999, None

                    for a in self.A:
                        v = self.R[tuple(s), a] + self.gamma * self.Sigma(tuple(s), a)
                        if v > max_v:
                            max_v, max_a = v, a

                    self.V[tuple(s)], self.Policy[tuple(s)] = max_v, max_a
                else:
                    if s in self.goal:
                        pass
                    else:
                        self.V[tuple(s)], self.Policy[tuple(s)] = 0.0, None


            V_current, V_last = self.Dict2Vec(self.V), self.Dict2Vec(self.V_)
            # print V_current, V_last
            it += 1
        # self.plot_curve(diff, "action diff")
        print diff
        self.action_diff = diff
        self.action_val = val
        self.action_pi_diff = pi_diff
        return 0

    def SVI_option(self, threshold):
        # self.init_V()
        # print self.R
        goal_room = None
        for opt in self.Opt.keys():
            if self.goal[0] in self.Opt[opt].S and self.goal[0] not in self.Opt[opt].wall_cord:
                goal_room = opt
        # print goal_room

        V_current, V_last = self.Dict2Vec(self.V), self.Dict2Vec(self.V_)
        # print V_current, V_last

        it = 1
        diff = []
        val = []
        pi_diff = []

        while np.inner(V_current - V_last, V_current - V_last) > threshold:
            if it > 24:
                break
            # print it, np.inner(V_current - V_last, V_current - V_last)
            # print "=================================================="
            val.append(np.linalg.norm(V_current))
            diff.append(np.inner(V_current - V_last, V_current - V_last))
            pi_diff.append(self.policy_diff(self.Policy, self.Policy_))

            self.plot("Option_result" + str(it), "../SMDPtools/HybridHeat/")

            for state in self.S:
                s = tuple(state)
                self.V_[s] = self.V[s]
                if s in self.Policy:
                    self.Policy_[s] = self.Policy[s]

            for s in self.S:

                if s not in self.wall_cord and s not in self.goal:
                    max_va, max_vo = -1.0 * 99999999, -1.0 * 99999999
                    max_a, max_opt = None, None

                    if s in self.Opt[goal_room].S and s not in self.Opt[goal_room].wall_cord:
                        for a in self.A:
                            v = self.R[tuple(s), a] + self.gamma * self.Sigma(tuple(s), a)
                            if v > max_va:
                                max_va, max_a = v, a

                        self.V[tuple(s)], self.Policy[tuple(s)] = max_va, max_a

                    else:
                        for opt in self.Opt.keys():
                            if s in self.Opt[opt].S and s not in self.Opt[opt].goal and s not in self.Opt[opt].wall_cord:
                                v = self.R[tuple(s), opt] + self.Sigma_opt(tuple(s), opt)
                                # print "optv", v
                                if v > max_vo:
                                    max_vo, max_opt = v, opt

                        self.V[tuple(s)], self.Policy[tuple(s)] = max_vo, max_opt

                    # if max_va > max_vo:
                    #     self.V[tuple(s)], self.Policy[tuple(s)] = max_va, max_a
                    # else:
                    #     self.V[tuple(s)], self.Policy[tuple(s)] = max_vo, max_opt
                else:
                    if s in self.goal:
                        pass
                    else:
                        self.V[tuple(s)], self.Policy[tuple(s)] = 0.0, None

            V_current, V_last = self.Dict2Vec(self.V), self.Dict2Vec(self.V_)
            # print V_current, V_last
            it += 1

            # print it, np.inner(V_current - V_last, V_current - V_last)
        # self.plot_curve(diff, "hybrid diff")
        # plt.plot(diff)
        # # plt.xlabel('episode')
        # plt.ylabel('value difference')
        # plt.show()
        # self.plot("Option_result")
        # print diff
        self.hybrid_diff = diff
        self.hybrid_val = val
        self.hybrid_pi_diff = pi_diff
        return 0

    def SVI(self, threshold, mode = "Act"):

        # TODO if choose "option" mode, switch to option directly after iteration reaches pathways
        # need classification: goal in room or right on the pathway?
            # if on pathway, no difference to the pure option based SVI
            # if inside room, do SVI in the room first

        #iteration: for states inside the room,
        if mode == "Act":
            self.SVI_action(threshold)
        elif mode == "Option":
            self.SVI_option(threshold)


        # self.plot("result")

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

    def plot(self, name, directory):

        temp = np.random.random((13, 13))
        for i in range(13):
            for j in range(13):

                if (i,j) not in self.V:
                    self.V[(i,j)] = -1

                temp[i, j] = self.V[(i, j)]

        plt.figure()
        plt.imshow(temp, cmap='hot', interpolation='nearest')
        plt.savefig(directory + name + ".png") #"../Gridworld/HybridHeat/
