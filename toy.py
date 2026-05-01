"""
Simple toy environment
"""
import torch as tr
import matplotlib.pyplot as pt

class ToyEnv:
    def __init__(self, init_stdev, delta_t):
        self.init_stdev = init_stdev # stdev in initial states
        self.delta_t = delta_t # time delta for forward euler

    def step(self, state, action):
        # action = tr.clamp(action, -2, 2)
        return tr.clamp(state + action * self.delta_t, 0, 1)
    
    def reward(self, states):
        return \
            tr.exp(-((states - tr.tensor([1.,1.]))**2).sum(dim=-1)/.2) - \
            tr.exp(-((states - tr.tensor([.4,.2]))**2).sum(dim=-1)/.1) - \
            tr.exp(-((states - tr.tensor([.6,.8]))**2).sum(dim=-1)/.1)    # return \
    
    def rollout(self, num_steps, batch_size, policy):
        states = [(self.init_stdev*tr.randn(batch_size, 2)).abs()]
        for t in range(num_steps-1):
            states.append(self.step(states[t], policy(t, states[t])) )
        return tr.stack(states)

    def contours(self, num_pts, levels):
        X, Y = tr.meshgrid(tr.linspace(0,1,num_pts),tr.linspace(0,1,num_pts))
        states = tr.stack([X.flatten(), Y.flatten()], axis=-1)
        R = self.reward(states).reshape(X.shape)
        pt.contourf(X, Y, R, levels=levels)
        pt.colorbar()

class TimeVaryingLinearPolicy:

    def __init__(self, num_steps):
        self.lins = [tr.nn.Linear(2,2) for _ in range(num_steps-1)]
        # init for straight diagonal trajectory
        for lin in self.lins:
            lin.weight.data[:] = 0.
            lin.bias.data[:] = 1.

    def __call__(self, t, states):
        return self.lins[t](states)

if __name__ == "__main__":

    init_stdev = 0.05
    num_steps = 30
    delta_t = 2**.5 / num_steps

    env = ToyEnv(init_stdev, delta_t)

    # X, Y = tr.meshgrid(tr.linspace(0,1,20),tr.linspace(0,1,20))
    # states = tr.stack([X.flatten(), Y.flatten()], axis=-1)
    # R = env.reward(states).reshape(X.shape)

    # do some rollouts
    def random_policy(t, states): return tr.randn(*states.shape)
    linear_policy = TimeVaryingLinearPolicy(num_steps)

    with tr.no_grad():
        rstates = env.rollout(num_steps, 1, random_policy)
        lstates = env.rollout(num_steps, 2, linear_policy)

    env.contours(num_pts=20, levels=20)
    # pt.contourf(X, Y, R, levels=20)
    # pt.colorbar()
    for states, color in [(rstates, 'b'), (lstates, 'r')]:
        for b in range(states.shape[1]):
            pt.plot(states[:,b,0], states[:,b,1], color+'.-')
    pt.show()
