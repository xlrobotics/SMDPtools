import numpy as np

class MDP:
    def __init__(self, S = [], A = [], P = {}, Pi = {}, R = {}, alpha = 0.0, gamma = 0.0, V = {}, Q = {}):
        self.S = [1, 2, 3]
        self.A = ['a', 'b']
        self.P = {}
        self.T = {}
        self.V = {}
        self.V_ = {}
        self.R = {}
        self.Memory = {}

        self.gamma = .9
        self.alpha = .5

    def Psa(self, s, a, s_, p):

        self.Tsa(s, a, s_)
        self.add_s(s)
        self.add_a(a)

        index = (s, a, s_)
        if index not in self.P:
            self.P[index] = p
        return 0

    def Tsa(self, s, a, s_):
        index = (s, a)
        if index not in self.T:
            self.T[index] = []

        if s_ not in self.T[index]:
            self.T[index].append(s_)
        return 0

    def add_s(self, s):
        if s not in self.S:
            self.S.append(s)
        return 0

    def add_a(self, a):
        if a not in self.A:
            self.A.append(a)

    def set_R(self, s, a, r):
        index = (s, a)
        if index not in self.R:
            self.R[index] = r
        return 0

    def init_V(self):
        for state in self.S:
            if state not in self.V:
                self.V[state] = 0.0
                self.V_[state] = 0.0

    # Vk[s] = max_a(R(s, a) + lambda *Vk-1[j] )
    def Dict2Vec(self, V):
        v = []
        for s in self.S:
            v.append(V[s])

        return np.array(v)

    def Sigma(self, s, a):
        index = (s, a)
        total = 0.0
        for s_ in self.T[index]:
            total += self.P[(s, a, s_)] * self.V_[s_]
        return total

    def SVI(self, threshold):
        self.init_V()

        for s in self.S:
            max_v = -1.0 * 99999999
            max_a = None
            for a in self.A:
                v = self.R[(s, a)] + self.gamma * self.Sigma(s, a)
                if v > max_v:
                    max_v, max_a = v, a

            self.V[s] = max_v
            self.Memory[s] = max_a

        V_current = self.Dict2Vec(self.V)
        V_last = self.Dict2Vec(self.V_)

        it = 1

        print V_current, V_last, np.inner(V_current - V_last, V_current - V_last)

        while np.inner(V_current - V_last, V_current - V_last) > threshold:

            for s in self.S:
                self.V_[s] = self.V[s]

            for s in self.S:
                max_v = -1.0 * 99999999
                max_a = None
                for a in self.A:
                    v = self.R[(s, a)] + self.gamma * self.Sigma(s, a)
                    if v > max_v:
                        max_v, max_a = v, a

                self.V[s] = max_v
                self.Memory[s] = max_a

            V_current = self.Dict2Vec(self.V)
            V_last = self.Dict2Vec(self.V_)

            print V_current, V_last

            it += 1

        print it

        return 0

    def VI(self):
        return 0

    def PI(self):
        return 0
    def LP(self):
        return 0
    def DLP(self):
        return 0


if __name__ == '__main__':
    model = MDP()

    model.Psa(1, 'a', 1, .7)
    model.Psa(1, 'a', 3, .3)
    model.Psa(1, 'b', 2, .6)
    model.Psa(1, 'b', 3, .4)

    model.set_R(1, 'a', .0)
    model.set_R(1, 'b', .2)

    model.Psa(2, 'a', 1, .3)
    model.Psa(2, 'a', 3, .7)
    model.Psa(2, 'b', 2, .5)
    model.Psa(2, 'b', 3, .5)

    model.set_R(2, 'a', .2)
    model.set_R(2, 'b', 1.)

    model.Psa(3, 'a', 3, 1.)
    model.Psa(3, 'b', 3, 1.)
    model.set_R(3, 'a', .0)
    model.set_R(3, 'b', .0)



    model.SVI(0.000000001)
    print model.V, model.V_
    print model.Memory