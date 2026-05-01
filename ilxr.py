import torch as tr

def ilxr_gd(env, policy, batch_size, num_updates, learning_rate):
    """
    Updates policy in place
    """

    reward_curve = []
    for update in range(num_updates):
        # collect rollouts
        with tr.no_grad():
            states = env.rollout(num_steps, batch_size, policy)
        rewards = env.reward(states)
        D = states.shape[-1] # state dimension

        net_reward = rewards.mean(dim=-1).sum()
        print(f"{update}: net reward = {net_reward:.5f}")
        reward_curve.append(net_reward)

        # fit linear models
        A, B, c, w, b = {}, {}, {}, {}, {}
        for t in range(num_steps-1):

            # dynamics
            with tr.no_grad():
                result = tr.linalg.lstsq(
                    tr.cat([states[t], policy(t, states[t]), tr.ones(batch_size,1)], dim=-1),
                    states[t+1])
            dyn = result.solution
            A[t], B[t], c[t] = dyn[:D], dyn[D:2*D], dyn[2*D:]

            # reward
            with tr.no_grad():
                result = tr.linalg.lstsq(
                    tr.cat([states[t+1], tr.ones(batch_size,1)], dim=-1),
                    rewards[t+1])
            rew = result.solution
            w[t+1], b[t+1] = rew[:D], rew[D:]

        # update policy
        objective = tr.tensor(0.)
        state = states[0]
        for t in range(num_steps-1):
            action = policy(t, state)
            state = state @ A[t] + action @ B[t] + c[t]
            reward = state @ w[t+1] + b[t+1]
            objective += reward.mean()

        objective.backward()
        for t in range(num_steps-1):
            policy.lins[t].weight.data += policy.lins[t].weight.grad * learning_rate
            policy.lins[t].bias.data += policy.lins[t].bias.grad * learning_rate
            policy.lins[t].weight.grad[:] = 0
            policy.lins[t].bias.grad[:] = 0

    return reward_curve

if __name__ == "__main__":

    import matplotlib.pyplot as pt
    from toy import ToyEnv, TimeVaryingLinearPolicy

    init_stdev = 0.05
    num_steps = 30
    delta_t = 2**.5 / num_steps
    batch_size = 64
    num_updates = 100
    learning_rate = .1

    env = ToyEnv(init_stdev, delta_t)
    policy = TimeVaryingLinearPolicy(num_steps)

    pt.figure(figsize=(12,4))

    pt.subplot(1,3,1)
    with tr.no_grad():
        states = env.rollout(num_steps, 3, policy)
    env.contours(num_pts=20, levels=20)
    for b in range(states.shape[1]):
        pt.plot(states[:,b,0], states[:,b,1], 'r.-')
    pt.title("Before training")

    reward_curve = ilxr_gd(env, policy, batch_size, num_updates, learning_rate)

    print(tr.stack([lin.bias.data for lin in policy.lins]))

    with tr.no_grad():
        states = env.rollout(num_steps, 3, policy)

    pt.subplot(1,3,2)
    with tr.no_grad():
        states = env.rollout(num_steps, 3, policy)
    env.contours(num_pts=20, levels=20)
    for b in range(states.shape[1]):
        pt.plot(states[:,b,0], states[:,b,1], 'r.-')
    pt.title("After training")

    pt.subplot(1,3,3)
    pt.plot(reward_curve)
    pt.xlabel("Gradient update")
    pt.ylabel("Net reward (batch average)")
    pt.title("Reward Curve")

    pt.tight_layout()
    pt.savefig("toy.png")
    pt.show()

