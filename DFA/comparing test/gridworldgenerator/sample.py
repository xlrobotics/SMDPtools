import numpy as np
from copy import deepcopy as dcp

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


global_g = 100.0
# transition probability setting P(s'|s,a)
# input:
# S: state space,
# wall_cord: boundary states,
# A: action dict,
# epsilon: 1-epsilon = transition probability for P(s+a|s,a)

# !!!the law of grid world is: P(s'|s,a) = 1-epsilon when s' = s+a, otherwise P(s'|s+a) = epsilon/3, if s' exceed
# the state space S, then P(s|s,a) += epsilon/3

def GeoSeries(P, it):
    sum = dcp(P)  # + np.identity(len(P))
    for k in range(2, it + 1):
        sum += np.linalg.matrix_power(dcp(P), k)
    # print P
    # print 50
    # print np.linalg.matrix_power(dcp(P), 50)
    # print 100
    # print np.linalg.matrix_power(dcp(P), 1000)
    # print 1000
    # print np.linalg.matrix_power(dcp(P), 1000)

    # sum = np.linalg.matrix_power(dcp(P), it)
    return sum

# API: calculating transition matrix for an option
def transition_matrix_(S, goal, unsafe, interruptions, gamma, P, Pi, row, col): # write the correct transition probability according to the policy
    H = {}
    tau = 1.0
    # row, col = 6, 8
    size  = row * col + 1 #l * w + sink
    PP = np.zeros((size, size))
    HH = np.zeros(size)
    PP[0, 0] = 1.0
    HH[0] = 1.0
    for state in S:
        s = tuple(state)
        n = (s[0] - 1) * col + s[1] #(1,1) = 1, (2,1)=9

        if s not in goal and s not in unsafe and s not in interruptions:
            PP[n, 0] = 1 - gamma
            for a in Pi[s]:
                HH[n] += -tau * Pi[s][a] * np.log(Pi[s][a])
                
                for nb in P[s, a]:
                    n_nb = (nb[0] - 1) * col + nb[1]
                    PP[n, n_nb] += P[s, a][nb] * Pi[s][a] * gamma


                    # if PP[n, n_nb] != 0:
                    #     HH[n] += -tau * PP[n, n_nb] * np.log(PP[n, n_nb])
        else:
            PP[n, n] = 1.0
            HH[n] = 0.0

    sums = GeoSeries(
        dcp(PP),
        100)
    sum_entropy = sums.dot(HH) *0.0
    # print 'size of', len(sum_entropy)
    # print type(sums)
    final = {}

    result = np.linalg.matrix_power(dcp(PP), 100)
    for state in S:
        s = tuple(state)
        n = (s[0]-1) * col + s[1]

        final[s] = {}
        line = []
        for state_ in S:
            s_ = tuple(state_)
            n_ = (s_[0]-1) * col + s_[1]
            line.append(result[n, n_])
        for g in goal:
            ng = (g[0]-1) * col + g[1]
            # final[s][g] = result[n, ng]/sum(line)
            final[s][g] = result[n, ng]

        H[s] = sum_entropy[n]

    print final


    return final, H


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
    for i in range(1, l+1):
        for j in range(1, w+1):
            inner.append([i, j])
    return inner

def Sigma(s, a, P, V_):
    total = 0.0

    for s_ in P[s, a].keys():
        if s_ != s:
            total += P[s, a][s_] * V_[s_]
    return total

def init_V(S, goal, g = 100.0, ng = None):
    V, V_ = {}, {}
    for state in S:
        s = tuple(state)
        if s not in V:
            V[s], V_[s] = 0.0, 0.0
        if s in goal:
            V[s], V_[s] = g, g
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

def Sigma_(s, goal, tx, Hs):
    result = Hs
    for g in goal:
        result += tx[s][g] * global_g

    return result

def Softmax_SVI(S, A, P, goal, ng, threshold, gamma, R, H, init = None):

    Pi = {}
    Q = {}
    V_record = []
    tau = 1

    if init == None:
        V, V_ = init_V(S, goal, global_g, ng)
    else:
        # V, V_ = dcp(init), dcp(init)
        V, V_ = init_V(S, goal, global_g, ng)

    V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)

    it = 1

    while it == 1 or np.inner(V_current - V_last, V_current - V_last) > threshold:  # np.inner(V_current - V_last, V_current - V_last) > threshold
        # print V

        V_record.append(V[(2, 2)])


        for s in S:
            V_[tuple(s)] = V[tuple(s)]

        # plot_heat("SVI_result" + str(it), V, 7, 9)


        for state in S:
            s = tuple(state)
            if s not in goal and s not in ng:
                if s in goal:
                    print s, goal
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
                        # Q[s][a] = np.exp((gamma * Sigma(s, a, P, V_)) / tau)
                        # Q[s][a] = np.exp((R[s][a] + gamma * V_[s_]) / tau)
                        Q[s][a] = np.exp((gamma * V_[s_]) / tau)
                        # print Q[tuple(s)]
                if init is not None:
                    Q[s]['opt'] = np.exp(Sigma_(s, goal, init, H[s]))


                Q_s = np.sum(Q[s].values())
                for a in A:
                    if (s, a) in P:
                        Pi[s][a] = Q[s][a] / Q_s

                if init is not None:
                    Pi[s]['opt'] = Q[s]['opt']/Q_s
                # print Q[tuple(s)].values()
                # V[s] = tau * np.log(np.dot(Q[s].values(), Pi[s].values())) # /len(Q[tuple(s)])
                V[s] = tau * np.log(Q_s)
            else:
                # print s, goal
                # V[s] = max(V.values())
                # print it, V[s]
                if s not in Pi:
                    Pi[s] = {}
                    for a in A:
                        Pi[s][a] = 0.0
                    if init is not None:
                        Pi[s]['opt'] = 0.0
                # Pi[tuple(s)] = []
                # pass
        # print V
        V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)
        it += 1
    print "softmax iteration num:", it

    return dcp(Pi), dcp(V), dcp(V_record)

def SVI(S, A, P, goal, threshold, gamma, R, init = None): #wall_cord,

    Pi = {}
    if init == None:
        V, V_ = init_V(S, goal)
    else:
        V, V_ = dcp(init), dcp(init)
        for state in S:
            s = tuple(state)
            if state in goal:
                V[s] = 1.0
            if V_[s] == 1.0:
                V[s] = 0.0
    # print V

    V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)

    it = 1

    while it == 1 or np.inner(V_current - V_last, V_current - V_last) > threshold: # np.inner(V_current - V_last, V_current - V_last) > threshold

        for s in S:
            V_[tuple(s)] = V[tuple(s)]

        # plot_heat("SVI_result" + str(it), V, 7, 9)
        # print V

        for state in S:
            s = tuple(state)

            if s not in goal:
                max_v, max_a = -1.0 * 99999999, None
                for a in A:
                    if (s, a) in P:
                        s_ = tuple(np.array(s) + np.array(A[a]))
                        if list(s_) not in S:
                            s_ = s
                        v = R[s][a] + gamma * Sigma(s, a, P, V_)
                        # v = R[s][a] + gamma * V_[s_]
                        if v > max_v:
                            max_v, max_a = v, a

                V[s], Pi[s] = max_v, max_a
            else:
                V[s] = max(V.values())
                print V[s], it
                Pi[s] = []

        V_current, V_last = Dict2Vec(V, S), Dict2Vec(V_, S)
        it += 1
    print it

    return dcp(Pi), dcp(V)

def DisjunctionApproximation(V1, V2, R1, R2, S, A, g, V, method = 'Hardmax'):
    result = {}
    tau = 1
    if method == 'Hardmax':
        for state in S:
            s = tuple(state)
            result[s] = max(V1[s], V2[s])
    elif method == 'Softmax':
        for state in S:
            s = tuple(state)
            smoothing  = 0.00000000000001
            Pv = np.exp(V1[s])/(np.exp(V1[s]) + np.exp(V2[s])) + smoothing
            Qv = np.exp(V2[s])/(np.exp(V1[s]) + np.exp(V2[s])) + smoothing
            # H = - (Pv * np.log(Qv) + Qv * np.log(Pv))

            # HH = 2 * Pv * Qv * (np.exp(V1[s]) + np.exp(V2[s]))

            # HH = np.exp(- Pv * Qv)

            N = np.exp(2*V1[s]) + np.exp(2*V2[s]) # + np.exp(2*H)
            D = np.exp(V1[s]/tau) + np.exp(V2[s]/tau) # + np.exp(H)
            # print V[s], np.log(D), V[s] - np.log(D)

            # HH = np.exp(- Pv * Qv)
            #
            HH = 1

            # print 'H value:', HH, N/D, state

            result[s] = tau * np.log(D)

            if s in g:
                if V1[s] != V2[s]:
                    result[s] = max(V1[s], V2[s])
                else:
                    result[s] = tau * np.log(D)

            # result[s] = np.log((np.exp(V1[s]) + np.exp(V2[s])) ** 2 / (
            # np.exp(V1[s]) + np.exp(V2[s]) + np.exp((np.log(2) + V1[s] + V2[s]) / 2)))

            # if V1[s] < V[s] and V2[s] < V[s]:
            #     result[s] = np.log((np.exp(V1[s]) + np.exp(V2[s]))**2/(np.exp(V1[s]) + np.exp(V2[s]) + np.exp((np.log(2)+V1[s]+V2[s])/2)))
            # else:
            #     result[s] = np.log()
                # result[s] = np.log(
                #     (np.square(np.exp(V1[s])) + np.square(np.exp(V2[s]))) / (np.exp(V1[s]) + np.exp(V2[s])))

            # if s in g:
            #     result[s] = 1.0
            # a = 'N'
            # for action in A:
            #     if R1[s][a] > 0 or R2[s][a] > 0:
            #         a = action
            #         break
            #
            # result[s] = np.log((np.exp(R1[s][a]) * np.exp(V1[s]) + np.exp(R2[s][a]) * np.exp(V2[s])) / (np.exp(R1[s][a]) + np.exp(R2[s][a])))
    elif method == 'linear':
        for state in S:
            s = tuple(state)
            result[s] = (V1[s] + V2[s]) / 100

    elif method == 'notuntil':
        for state in S:
            s = tuple(state)
            result[s] = V1[s] - V2[s]

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
            # result[s] = np.log(abs(np.exp(A[s]) - np.exp(B[s])))
            '''
            if A[s] > B[s]:
                result[s] = 1
            elif A[s] == B[s]:
                result[s] = 0
            else:
                result[s] = -1
            '''
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

    temp = np.random.random((l, w))
    for i in range(l):
        for j in range(w):
            s = tuple([i+1, j+1])
            if s in V:
                temp[i,j] = V[s]
            else:
                temp[i,j] = -1

    # plt.figure()
    # x, y = np.mgrid[-1.0:1.0:19j, -1.0:1.0:19j]
    # # #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot_wireframe(x, y, temp)
    # surf = ax.plot_surface(x, y, temp)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # plt.show()
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # # Make data.
    # X = np.arange(0, l-1, 1)
    # Y = np.arange(0, w-1, 1)
    # X, Y = np.meshgrid(X, Y)
    # Z = temp
    #
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    cax = plt.imshow(temp, cmap='hot', interpolation='nearest')
    fig.colorbar(cax)
    plt.savefig(name + ".png") # "../DFA/comparing test/"

def plot_traj(name, V, l, w, traj): #dir,
    temp = np.random.random((l, w))
    for i in range(l):
        for j in range(w):
            s = tuple([i, j])
            if s in V:
                if s in traj:
                    temp[s] = V[s]
                else:
                    temp[s] = -1
            else:
                temp[s] = -1

    x, y = np.mgrid[-1.0:1.0:l, -1.0:1.0:w]

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.plot_wireframe(x, y, temp)
    # ax.plot_surface(x, y, temp)
    # plt.show()
    plt.imshow(temp, cmap='hot', interpolation='nearest')
    plt.savefig(name + ".png")  # "../DFA/comparing test/"

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

def Simple_regression(V,V1,V2):
    W1, W2 = {}, {}
    V_approx = {}
    threshold = 0.0000001
    V_mark = {}
    alpha = 0.01
    for s in V:
        W1[s] = np.exp(V1[s])/(np.exp(V1[s]) + np.exp(V2[s]))
        W2[s] = 1 - W1[s]
        V_approx[s] = W1[s] * V1[s] + W2[s] * V2[s]

        diff = V_approx[s] - V[s]
        # print s, diff, V[s], V1[s], V2[s]

        if V1[s] > V[s] and V2[s] > V[s]:
            V_mark[s] = 1.0
        elif V1[s] < V[s] and V2[s] < V[s]:
            V_mark[s] = -1
        else:
            V_mark[s] = 0
        '''
        it = 1
        while abs(diff) > threshold:
            # if V1[s] >= V[s] and V2[s] >= V[s]:
            #     break
            # if V1[s] <= V[s] and V2[s] <= V[s]:
            #     break
            if it > 100:
                print "time out"
                break

            diff = V_approx[s] - V[s]

            # print diff, V[s], V1[s], V2[s]

            if V1[s] > V2[s]:
                W1[s] -= 100 * diff
                W2[s] = 1 - W1[s]
            else:
                W2[s] -= 100 * diff
                W1[s] = 1 - W2[s]
            V_approx[s] = W1[s] * V1[s] + W2[s] * V2[s]
            it += 1
        '''

    return W1, W2, V_approx, V_mark

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

def entropyMap(V1, V2, S, gamma, A):
    Ve = {}
    Vf = {}
    for state in S:
        s = tuple(state)
        smoothing = 0
        Pv = np.exp(V1[s]) / (np.exp(V1[s]) + np.exp(V2[s])) + smoothing
        Qv = np.exp(V2[s]) / (np.exp(V1[s]) + np.exp(V2[s])) + smoothing
        # H = - (Pv * np.log(Qv) + Qv * np.log(Pv))
        # H = (Pv * Qv)
        H_hat = np.exp(V1[s]) + np.exp(V2[s])
        # H_hat = (Pv + Qv) ** gamma
        H_tilda = np.exp(gamma * V1[s]) + np.exp(gamma * V2[s])
        # H_tilda = H_hat ** gamma
        # H_tilda = Pv**gamma + Qv**gamma

        Ve[s] = np.log(H_hat)

    for state in S:
        s = tuple(state)
        Vf[s] = 0
        for a in A:
            s_ = tuple(np.array(s) + np.array(A[a]))
            if list(s_) not in S:
                s_ = s
            Vf[s] += np.exp(gamma * Ve[s_])

    for state in S:
        s = tuple(state)
        # Vf[s] = abs(Vf[s] - np.exp(Ve[s]))
        Vf[s] = Ve[s] - np.log(Vf[s])


    return Vf

def plot_curve(trace1, trace2, trace, name):
    plt.figure()

    # print x
    # print trace1
    # print trace2
    l2, = plt.plot(trace2, label="V2")
    l1, = plt.plot(trace1, label="V1")

    l, = plt.plot(trace, label="V")

    plt.legend(handles=[l1, l2, l])

    plt.xlabel('Value iteration episode')
    plt.ylabel('Value iteration function at (9, 11)')
    plt.show()
    # plt.savefig(name + ".png")

def Sigma_opt(S, goal, tx, H):
    V = {}
    for state in S:
        s = tuple(state)
        V[s] = 0
        for g in goal:
            V[s] += tx[s][g] * global_g
        V[s] += H[s]
    return V

def policyEvaluation(Pi, Pi_opt, S, A):
    Pi_tilda = {}
    for state in S:
        s = tuple(state)
        Pi_tilda[s] = {}
        po = Pi_opt[s]['opt']
        # TODO this part is unclear!!!!!
        for a in A:
            Pi_tilda[s][a] = Pi[s][a] * po + Pi_opt[s][a]

    return Pi_tilda

if __name__ == '__main__':
    l, w = 20, 20
    S = set_S(l, w) # set the width and length of grid world

    print S
    A = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)} # initialize action space
    epsilon = 0.3 # define transition probability: 1-epsilon = P(s+a|s,a)
    gamma = 0.9

    P = set_P(S, A, epsilon) #w
    print "Transition Probablilities:"
    for key in P:
        print key, P[key]
    print "===================================================="

    a = "N"
    threshold = 0.00000001


    goal1 = [(10, 10)] # (10, 11), (20, 19),
    # goal2 = [(10, 10),(10, 11)] # (10, 11)
    # print
    # goal = list(set(goal1 + goal2))  # set the end point of your trajectory
    #
    R1 = init_R(S, goal1, P, A)
    # R2 = init_R(S, goal2, P, A)
    # R = add_R(R1, R2, S, A)
    # print R1
    # print R2

    # Pi, V, traj = Softmax_SVI(S, A, P, goal, [], threshold, gamma, R)


    # Pi, V = SVI(S, A, P, goal, threshold, gamma, R) # generate policy based on synchronous value iteration (SVI)
    # Pi, V = Softmax_SVI(S, A, P, goal, threshold, gamma, R)
    # print V
    # v = extractPi(Pi, S, a)
    # v = ShowPolicy(Pi, S)
    # for key in V:
    #     V[key] = np.exp(V[key]-6)
    # V[(1,1)], V[(6,7)] = 1.0, 1.0

    # goal = [[6, 6]]  # set the end point of your trajectory
    # Pi, V = SVI(S, A, P, goal, 0.01, 0.9, V)
    # goal = [[5, 5]]
    # Pi, V = SVI(S, A, P, goal, 0.01, 0.9)
    # V is the iteration value of state space for vision aid
    # print Pi[(1, 1)], Pi[(1, 2)], Pi[(2, 2)]


    # Pi1, V1 = SVI(S, A, P, goal1, threshold, gamma, R1)

    # global_g = 13.8629436112
    # global_g = -100
    # global_g = 100
    # threshold = 1000000000
    zeroH = {}
    for state in S:
        s = tuple(state)
        zeroH[s] = 0

    Pi1, V1, traj1 = Softmax_SVI(S, A, P, goal1, [], threshold, gamma, R1, zeroH)

    defaultP = {}
    for state in S:
        s = tuple(state)
        for a in A:
            defaultP[s, a] = {}
            s_ = tuple(np.array(s) + np.array(A[a]))
            if list(s_) not in S:
                s_ = s
            defaultP[s, a][s_] = 1.0

    TM, H = transition_matrix_(S, goal1, [], [], gamma, defaultP, Pi1, l, w)
    # print H

    # print len(H[1,1])

    Pi_opt, V_opt, traj_opt = Softmax_SVI(S, A, P, goal1, [], threshold, gamma, R1, H, init=TM)

    Pi_tilda = policyEvaluation(Pi1, Pi_opt, S, A)

    V_opt_plain = Sigma_opt(S, goal1, TM, H)


    # global_g = 95
    # Pi1, V11, traj1 = Softmax_SVI(S, A, P, goal1, [], threshold, gamma, R1)
    # global_g = 90
    # Pi1, V111, traj1 = Softmax_SVI(S, A, P, goal1, [], threshold, gamma, R1)
    # global_g = 85
    # Pi1, V1111, traj1 = Softmax_SVI(S, A, P, goal1, [], threshold, gamma, R1)
    #
    # V_diff1 = VDiff(V1, V11, V1, V1, S)
    # V_diff11 = VDiff(V11, V111, V1, V1, S)
    # V_diff111 = VDiff(V111, V1111, V1, V1, S)
    # for g in goal1:
    #     V1[g] = max(V1.values())

    # print max(V1.values()), min(V1.values())
    # print V1
    # print V_diff1
    # print V_diff11
    # print V_diff111

    # plot_heat("SVI_result_diff1", V_diff1, l, w)
    # plot_heat("SVI_result_diff11", V_diff11, l, w)
    # plot_heat("SVI_result_diff111", V_diff111, l, w)
    # nH = {}
    # for key in H:
    #     nH[key] = -H[key]

    plot_heat("SVI_result_V1", V1, l, w)
    plot_heat("Option_V1", V_opt, l, w)
    plot_heat("Option_V1_plain", V_opt_plain, l, w)
    plot_heat("entropy", H, l, w)
    print V1
    print V_opt
    print V_opt_plain


    print Pi_opt
    print Pi_tilda
    print Pi1
    # for s in goal1:
    #     V1[s] = 0
    # v1 = extractPi(Pi1, S, a)
    # v1 = ShowPolicy(Pi1, S)
    # Pi1, V1 = Softmax_SVI(S, A, P, goal1, threshold, gamma, R1)
    # print V1
    # V1[(1, 1)] = 1.0
    # for key in V1:
    #     V1[key] /= 100.0
    # Pi2, V2 = SVI(S, A, P, goal2, threshold, gamma, R2)

    # Pi2, V2, traj2 = Softmax_SVI(S, A, P, goal2, [], threshold, gamma, R2)

    # for s in goal2:
    #     V2[s] = 0

    print "================================================"
    # R1, R2, V1, V2 = rewardModify(R1, R2, V1, V2, S, A)
    #
    # Pi1, V1 = Softmax_SVI(S, A, P, goal1, [], threshold, gamma, R1, V1)
    # #
    # Pi2, V2 = Softmax_SVI(S, A, P, goal2, [], threshold, gamma, R2, V2)


    # v2 = extractPi(Pi2, S, a)
    # v2 = ShowPolicy(Pi2, S)
    # Pi2, V2 = Softmax_SVI(S, A, P, goal2, threshold, gamma, R2)
    # print V2
    # for key in V2:
    #     V2[key] /= 100.0
    # V2[(6,7)] = 1.0
    #
    # V_hat = V

    # V_tilda = DisjunctionApproximation(V1, V2, R1, R2, S, A, goal, V, 'Softmax') #'Softmax'

    # V_tilda = V1

    # for s in S:
    #     V[tuple(s)] *= 1000
    #     V1[tuple(s)] *= 1000
    #     V2[tuple(s)] *= 1000
    #     V_tilda[tuple(s)] *= 1000
    # Pi_tilda, V_tilda = Softmax_SVI(S, A, P, goal, [], threshold, gamma, R, V_tilda)

    # V_tilda[5,5], V_tilda[5,14] = 0.0, 0.0


    # print V_tilda
    # Pi_tilda, V_tilda = Softmax_SVI(S, A, P, goal, threshold, gamma, R, V_tilda)
    # _, _ = Softmax_SVI(S, A, P, goal, threshold, gamma, R, V1)
    # _, _ = Softmax_SVI(S, A, P, goal, threshold, gamma, R, V2)
    # _, _ = Softmax_SVI(S, A, P, goal, threshold, gamma, R, V_tilda)

    # Pi_tilda, _ = Softmax_SVI(S, A, P, goal, 100000, gamma, R, V_tilda)
    # v_tilda = extractPi(Pi_tilda, S, a)

    # v_tilda = ShowPolicy(Pi_tilda, S)
    # TODO: add one more iteration to get the desired policy

    # V_tilda = DisjunctionApproximation(V1, V2, S, 'Softmax')
    # print Pi_tilda
    # print v_tilda
    # for key in V_tilda:
    #     V_tilda[key] = np.exp(V_tilda[key]-6)



    # v_hat = v
    #
    # V_diff = VDiff(V_tilda, V_hat, V1, V2, S)

    # V_diff = VDiff(V_tilda, V_hat, V1, V2, S)

    # V_diff1 = VDiff(V1, V_hat, S, 'Softmax')
    # V_diff2 = VDiff(V2, V_hat, S, 'Softmax')
    # V_entropy = entropyMap(V1, V2, S, gamma, A)


    # v_diff = VDiff(v_tilda, v_hat, S)
    # V_diff = VDiff(V_tilda, V_hat, S, 'Softmax')

    # print V1
    # print V2
    # print V_hat
    # print V_tilda
    # print V_diff

    # print V_entropy
    # v_diff = VDiff(v_tilda, v_hat, S, 'Softmax')

    # for key in V_diff:
    #     V_diff[key] = 100.0 * abs(V_diff[key] - 0.693)
    #     if list(key) in goal:
    #         V_diff[key] = 0

    # print V_diff
    # print v_diff

    # w1, w2, V_approx, V_mark = Simple_regression(V, V1, V2)
    # V_diff2 = VDiff(V_approx, V_hat, V1, V2, S, 'Softmax')

    '''
    plot_heat("SVI_result_true", V, l, w)
    plot_heat("SVI_result_goal1", V1, l, w)
    plot_heat("SVI_result_goal2", V2, l, w)
    plot_heat("SVI_result_approximated", V_tilda, l, w)
    plot_heat("SVI_result_diff", V_diff, l, w)
    '''

    # plot_heat("SVI_result_entropy", V_entropy, l, w)
    # plot_curve(traj1, traj2, traj, 'compare_9_11')

    # plot_heat("SVI_result_diff1", V_diff1, l, w)
    # plot_heat("SVI_result_diff2", V_diff2, l, w)
    # plot_heat("SVI_result_w1", w1, l, w)
    # plot_heat("SVI_result_w2", w2, l, w)
    # plot_heat("SVI_result_mark", V_mark, l, w)

    # print V_diff
    # plot_heat("SVI_result_true", v, 17, 19)
    # plot_heat("SVI_result_goal1", v1, 17, 19)
    # plot_heat("SVI_result_goal2", v2, 17, 19)
    # plot_heat("SVI_result_approximated", v_tilda, 17, 19)
    # plot_heat("SVI_result_diff", v_diff, 17, 19)

    # start = [1, 1] # set the start point of your trajectory
    # traj = GenTraj(Pi, start, A, V) # generate trajectory
    #
    # plot_traj("traj", V, 7, 9, traj)
    # print "trajectory start from", start, "to", goal, ":"
    # print traj


