import numpy as np
from copy import deepcopy as dcp

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.mplot3d import Axes3D

# transition probability setting P(s'|s,a)
# input:
# S: state space,
# wall_cord: boundary states,
# A: action dict,
# epsilon: 1-epsilon = transition probability for P(s+a|s,a)

# !!!the law of grid world is: P(s'|s,a) = 1-epsilon when s' = s+a, otherwise P(s'|s+a) = epsilon/3, if s' exceed
# the state space S, then P(s|s,a) += epsilon/3

def set_P(S, A, epsilon): #wall_cord,
    P = {}
    for state in S:
        s = tuple(state)
        explore = []

        for act in A.keys():
            temp = tuple(np.array(s) + np.array(A[act]))
            explore.append(temp)
        # print s, explore
        for a in A.keys():
            P[s, a] = {}
            P[s, a][s] = 0

            s_ = tuple(np.array(s) + np.array(A[a]))
            unit = epsilon / 3

            if list(s_) in S:

                P[s, a][s_] = 1 - epsilon
                for _s_ in explore:
                    if tuple(_s_) != s_:
                        if list(_s_) in S:
                            P[s, a][tuple(_s_)] = unit
                        else:
                            P[s, a][s] += unit
            else:
                P[s, a][s] = 1 - epsilon
                for _s_ in explore:
                    if _s_ != s_:
                        if list(_s_) in S:
                            P[s, a][tuple(_s_)] = unit
                        else:
                            P[s, a][s] += unit

    return dcp(P)

def set_S(l, w):
    inner = []
    for i in range(1, l):
        for j in range(1, w):
            inner.append([i, j])
    return inner

def Sigma(s, a, P, V_):
    total = 0.0

    for s_ in P[s, a].keys():
        if s_ != s:
            total += P[s, a][s_] * V_[s_]
    return total

def init_V(S, goal, ng = None):
    V, V_ = {}, {}
    for state in S:
        s = tuple(state)
        if s not in V:
            V[s], V_[s] = 0.0, 0.0
        if s in goal:
            V[s], V_[s] = 100.0, 100.0
        if ng != None and s in ng:
            V[s], V_[s] = 0.0, 0.0
    return dcp(V), dcp(V_)

def init_R(S, goal, P, A, ng = None):
    R = {}
    g = 100.0

    # v = np.log((np.exp(0) * np.exp(0) + np.exp(g) * np.exp(g)) / (np.exp(0) + np.exp(g))) if len(goal) > 1 else g
    v = g
    nongoal = []
    if ng != None:
        nongoal = ng

    for state in S:
        s = tuple(state)
        if s not in R:
            R[s] = {}
        for a in A:
            if (s, a) in P:
                s_ = tuple(np.array(s) + np.array(A[a]))
                if s_ in goal:
                    R[s][a] = v
                elif s_ in nongoal:
                    R[s][a] = 0.0
                else:
                    R[s][a] = 0.0

    return dcp(R)

def add_R(R1, R2, S, A):
    R = {}
    for state in S:
        s = tuple(state)
        if s not in R:
            R[s] = {}
        for a in A:
            R[s][a] = R1[s][a] + R2[s][a]
            if R1[s][a] > 0 and R2[s][a] > 0:
                R[s][a] /= 2
    return dcp(R)

def Dict2Vec(V, S):
    v = []
    for s in S:
        v.append(V[tuple(s)])
    return np.array(v)

def Softmax_SVI(S, A, P, goal, ng, threshold, gamma, R = None, init = None):

    Pi = {}
    Q = {}
    V_record = []
    tau = 1
    if init == None:
        V, V_ = init_V(S, goal, ng)
    else:
        V, V_ = dcp(init), dcp(init)

    V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)

    it = 1

    while it == 1 or np.inner(V_current - V_last,
                   V_current - V_last) > threshold:  # np.inner(V_current - V_last, V_current - V_last) > threshold


        V_record.append(sum(V.values()))


        for s in S:
            V_[tuple(s)] = V[tuple(s)]

        # plot_heat("SVI_result" + str(it), V, 7, 9)

        for state in S:
            s = tuple(state)
            if s not in goal and s not in ng:

                # max_v, max_a = -1.0 * 99999999, None
                if s not in Pi:
                    Pi[s] = {}
                if s not in Q:
                    Q[s] = {}

                for a in A:
                    if (s, a) in P:
                        s_ = tuple(np.array(s) + np.array(A[a]))
                        if list(s_) not in S:
                            s_ = s
                            # Q[s][a] = 0
                            # continue
                        # Q[s][a] = np.exp((R[s][a] + gamma * Sigma(s, a, P, V_)) / tau)
                        # Q[s][a] = np.exp((R[s][a] + gamma * V_[s_]) / tau)
                        Q[s][a] = np.exp((gamma * V_[s_]) / tau)
                        # print Q[tuple(s)]
                Q_s = np.sum(Q[s].values())
                for a in A:
                    if (s, a) in P:
                        Pi[s][a] = Q[s][a] / Q_s
                # print Q[tuple(s)].values()
                # V[s] = tau * np.log(np.dot(Q[s].values(), Pi[s].values())) # /len(Q[tuple(s)])
                V[s] = tau * np.log(Q_s)
            else:
                # print s, goal
                # V[s] = 0
                if s not in Pi:
                    Pi[s] = {}
                    for a in A:
                        Pi[s][a] = 0.0
                # Pi[tuple(s)] = []
                # pass
        # print V
        V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)
        it += 1
    print "softmax iteration num:", it

    return dcp(Pi), dcp(V), dcp(V_record)

def DisjunctionApproximation(V_list, S, A, g, V, method = 'Hardmax'):
    result = {}
    tau = 1
    if method == 'Hardmax':
        for state in S:
            s = tuple(state)
            result[s] = 0
            for V_temp in V_list:
                if V_temp[s] > result[s]:
                    result[s] = V_temp[s]

    elif method == 'Softmax':
        for state in S:
            s = tuple(state)

            # N = np.exp(2*V1[s]) + np.exp(2*V2[s]) # + np.exp(2*H)
            # D = np.exp(V1[s]/tau) + np.exp(V2[s]/tau) # + np.exp(H)
            D = 0

            for V_temp in V_list:
                D += np.exp(V_temp[s]/tau)

            weighted_total = 0
            for V_temp in V_list:
                weighted_total += np.exp(V_temp[s]/tau)**2/D


            # result[s] = tau * np.log(weighted_total)
            result[s] = tau * np.log(D)

            if s in g:
                result[s] = 100.0

                # if V1[s] != V2[s]:
                #     result[s] = max(V1[s], V2[s])
                # else:
                #     result[s] = tau * np.log(D)

    elif method == 'linear':
        for state in S:
            s = tuple(state)
            for V_temp in V_list:
                result[s] += V_temp[s]

    # elif method == 'notuntil':
    #     for state in S:
    #         s = tuple(state)
    #         result[s] = V1[s] - V2[s]

    return dcp(result)

def VDiff(A, B, V1, V2, S, method = 'Hardmax'):
    result = {}
    for state in S:
        s = tuple(state)
        if method == 'Hardmax':
            # print A
            # print B
            result[s] = abs(A[s] - B[s])
        elif method == 'Softmax':

            B[s] += 0.00000000000001
            if A[s] == B[s]:
                result[s] = np.exp(1)
            else:
                result[s] = np.exp(1 - A[s]/B[s])

            if result[s] < 2.65:
                print s, A[s], B[s], V1[s], V2[s]

            # result[s] = 0 if A[s] == B[s] else abs(np.exp(A[s] - B[s]))
        # result[tuple(s)] = abs(np.exp(A[tuple(s)]) - np.exp(B[tuple(s)]))
    return dcp(result)

def GenTraj(Pi, start, A, V):
    traj = []
    current = start
    traj.append(current)
    a = Pi[tuple(current)]
    while a!= []:
        # print current, a
        current = tuple(np.array(current) + np.array(A[a]))
        traj.append(current)
        a = Pi[tuple(current)]

    return traj

def plot_heat(name, V, l, w):

    temp = np.random.random((l-1, w-1))
    for i in range(l-1):
        for j in range(w-1):
            s = tuple([i+1, j+1])
            if s in V:
                temp[i,j] = V[s]
            else:
                temp[i,j] = -1

    plt.figure()

    plt.imshow(temp, cmap='hot', interpolation='nearest')
    plt.savefig(name + ".png") # "../DFA/comparing test/"

def extractPi(Pi, S, a):
    print Pi
    value = {}
    for state in S:
        s = tuple(state)
        value[s] = Pi[s][a]
    return dcp(value)

def ShowPolicy(Pi, S):
    value = {}
    Action_mark = {"N":1.0, "S":2.0, "E":3.0, "W":4.0 }
    for state in S:
        s = tuple(state)
        if s not in value:
            value[s] = 0.0
        temp_max = 0.0
        for a in Action_mark:
            if Pi[s][a] > temp_max:
                value[s] = Pi[s][a]
                # value[s] = Action_mark[a]
                temp_max = Pi[s][a]
    return value

def rewardModify(R1, R2, V1, V2, S, A):
    mR1, mR2, mV1, mV2 = {}, {}, {}, {}

    for state in S:
        s = tuple(state)
        if V1[s] == V2[s] and V1[s] == 100:
            mV1[s], mV2[s] = np.log(0.5*np.exp(V1[s])), np.log(0.5*np.exp(V1[s]))
        else:
            mV1[s], mV2[s] = V1[s], V2[s]

        if s not in mR1:
            mR1[s], mR2[s] = {}, {}
        for a in A:
            if R1[s][a] == R2[s][a]:
                value = np.log(2*np.exp(R1[s][a]))
                mR1[s][a], mR2[s][a] = value, value
            else:
                mR1[s][a], mR2[s][a] = R1[s][a], R2[s][a]
    return mR1, mR2, mV1, mV2

def plot_curve(trace_list, approx_trace, trace, name):
    plt.figure()
    print approx_trace
    # print x
    # print trace1
    # print trace2
    # l2, = plt.plot(trace2, label="V2")
    # l1, = plt.plot(trace1, label="V1")
    l_list = [None] * len(trace_list)
    for i in range(len(trace_list)):
        l_list[i],  = plt.plot(trace_list[i], label="V"+str(i))

    l, = plt.plot(trace, label="V")
    l_app, = plt.plot(approx_trace, label="V_tilda")

    plt.legend(handles=l_list + [l_app, l])

    plt.xlabel('Value iteration episode')
    plt.ylabel('Value iteration function at (9, 11)')
    plt.show()
    # plt.savefig(name + ".png")

def read_P():
    P = {}

if __name__ == '__main__':
    l, w = 4, 4
    S = set_S(l, w) # set the width and length of grid world

    print S
    A = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)} # initialize action space
    epsilon = 0.3 # define transition probability: 1-epsilon = P(s+a|s,a)
    gamma = 0.9

    P = set_P(S, A, epsilon) #w
    P = read_P()
    print "Transition Probablilities:"
    for key in P:
        print key, P[key]
    print "===================================================="

    a = "N"
    threshold = 0.000001


    # goal1 = [(5, 5), (5, 6)] # (10, 11)
    # goal2 = [(5, 6), (5, 7)] # (10, 11)
    # goal3 = [(5, 7), (5, 8)]
    # goal4 = [(5, 8), (5, 9)]
    # goal5 = [(5, 9), (5, 10)]
    # goal6 = [(5, 10), (5, 11)]
    # goal7 = [(5, 11), (5, 12)]

    goal1 = [(5, 5)]  # (10, 11)
    goal2 = [(5, 6)]  # (10, 11)
    goal3 = [(5, 7)]
    goal4 = [(5, 8)]
    goal5 = [(5, 9)]
    goal6 = [(5, 10)]
    goal7 = [(5, 11)]
    goal8 = [(5, 12)]
    goal9 = [(5, 13)]
    goal10 = [(5, 14)]
    # print
    goal = list(set(goal1 + goal2))  # set the end point of your trajectory + goal3 + goal4

    Pi, V, traj = Softmax_SVI(S, A, P, goal, [], threshold, gamma)

    Pi1, V1, traj1 = Softmax_SVI(S, A, P, goal1, [], threshold, gamma)
    Pi2, V2, traj2 = Softmax_SVI(S, A, P, goal2, [], threshold, gamma)
    Pi3, V3, traj3 = Softmax_SVI(S, A, P, goal3, [], threshold, gamma)
    Pi4, V4, traj4 = Softmax_SVI(S, A, P, goal4, [], threshold, gamma)
    Pi5, V5, traj5 = Softmax_SVI(S, A, P, goal5, [], threshold, gamma)
    Pi6, V6, traj6 = Softmax_SVI(S, A, P, goal6, [], threshold, gamma)
    Pi7, V7, traj7 = Softmax_SVI(S, A, P, goal7, [], threshold, gamma)
    Pi8, V8, traj8 = Softmax_SVI(S, A, P, goal8, [], threshold, gamma)
    Pi9, V9, traj9 = Softmax_SVI(S, A, P, goal9, [], threshold, gamma)
    Pi10, V10, traj10 = Softmax_SVI(S, A, P, goal10, [], threshold, gamma)


    print "================================================"
    V_hat = V
    V_list  = [V1, V2] #V3, V4
    V_tilda = DisjunctionApproximation(V_list, S, A, goal, V, 'Softmax') #'Softmax'


    V_diff = VDiff(V_tilda, V_hat, V1, V2, S)
    # print V1
    # print V2
    # print V_hat
    # print V_tilda
    # print V_diff
    # print len(traj1), len(traj2)
    print np.array(V_diff.values()).dot(np.array(V_diff.values()))

    traj_approx = [None] * len(traj2)
    for i in range(len(traj2)):
        traj_approx[i] = traj1[i] + traj2[i]

    traj_list = [traj1, traj2]

    # plot_curve(traj_list, traj_approx, traj, 'compare_9_11')
    print np.array(V_diff.values()).dot(np.array(V_diff.values()))