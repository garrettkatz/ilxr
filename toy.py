"""
Simple toy environment
"""
import numpy as np

S0 = .01 # stdev in initial states and transitions
DT = .1 # time delta for forward euler

def step(state, action):
    return np.clip(state + action * DT + S0*np.random.randn(*state.shape), 0, 1)

def reward(states):
    return \
        np.exp(-((states - np.array([1.,1.]))**2).sum(axis=-1)/.2) - \
        np.exp(-((states - np.array([.4,.2]))**2).sum(axis=-1)/.1) - \
        np.exp(-((states - np.array([.6,.8]))**2).sum(axis=-1)/.1)

def rollout(num_steps, batch_size, policy):
    states = np.empty((num_steps, batch_size, 2))
    states[0] = np.fabs(S0*np.random.randn(batch_size, 2))
    for t in range(num_steps-1):
        states[t+1] = step(states[t], policy(t, states[t]))
    return states

class LinearPolicy:
    def __init__(self, num_steps):
        self.weights = [np.zeros((2,2)) for _ in range(num_steps)]
        self.bias = [np.ones(2) for _ in range(num_steps)]
    def __call__(self, t, states):
        return states @ self.weights[t] + self.bias[t]

if __name__ == "__main__":

    num_steps = 30

    X, Y = np.meshgrid(np.linspace(0,1,20),np.linspace(0,1,20))
    states = np.stack([X.flatten(), Y.flatten()], axis=-1)
    R = reward(states).reshape(X.shape)

    # do some rollouts
    def random_policy(t, states):
        return np.random.randn(*states.shape)
    linear_policy = LinearPolicy(num_steps)

    rstates = rollout(num_steps, 1, random_policy)
    lstates = rollout(num_steps, 2, linear_policy)

    import matplotlib.pyplot as pt
    pt.contourf(X, Y, R, levels=20)
    pt.colorbar()
    for states, color in [(rstates, 'b'), (lstates, 'r')]:
        for b in range(states.shape[1]):
            pt.plot(states[:,b,0], states[:,b,1], color+'.-')
    pt.show()
