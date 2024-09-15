import gymnasium as gym
import numpy as np
from tqdm import tqdm
import pickle

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 1
NUM_EPISODES = 500

def sample_action(policy, state):
    return np.random.choice(policy.shape[-1], p=policy[state])

def sarsa_control(s,a,r,s_prime, a_prime, q_s_a, policy):
    q_s_a[s,a] = q_s_a[s,a] + ALPHA*(r + GAMMA*q_s_a[s_prime, a_prime] - q_s_a[s,a])

    a_star = np.argmax(q_s_a[s])

    for action in range(policy.shape[-1]):
        if action == a_star:
            policy[s,action] = 1 - EPSILON + EPSILON/policy.shape[-1]
        else:
            policy[s,action] = EPSILON/policy.shape[-1]
    
    return q_s_a, policy


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')


    state_action_values = np.zeros((env.observation_space.n, env.action_space.n))   # (48, 4)
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

    if not np.all(policy.sum(-1) == 1):
        raise Exception("Action probabilities do not sum up to one!")

    episode_reward_list = []
    for episode in tqdm(range(NUM_EPISODES)):

        observation, info = env.reset()
        action = sample_action(policy, observation)
        terminated = False

        episode_reward = 0
        steps = 0
        while terminated == False:

            observation_prime, reward, terminated, truncated, info = env.step(action)
            action_prime = sample_action(policy, observation_prime)
            steps += 1
            #print(steps)
            
            state_action_values, policy = sarsa_control(
                s=observation,
                a=action,
                r=reward,
                s_prime=observation_prime,
                a_prime=action_prime,
                q_s_a=state_action_values,
                policy=policy
            )
            
            action = action_prime
            observation = observation_prime

            episode_reward += reward

            if reward == -100:
                terminated = True

        episode_reward_list.append(episode_reward)


    env.close()

    agent = {
        "policy": policy,
        "action_values": state_action_values,
        "ep_reward": episode_reward_list
    }

    save_object(agent, f"TD_SARSA_ep_{NUM_EPISODES}_eps_{EPSILON}_gam_{GAMMA}_alph_{ALPHA}.pkl")