import gymnasium as gym
import numpy as np
from tqdm import tqdm

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle




class OnPolicyFirstVistMCSoft():
    def __init__(self, observation_space, action_space) -> None:
        """
        Epsilon-soft Policy
        """
        state_action_shape =  [s.n for s in observation_space] + [action_space.n]

        # (player current sum, dealer showing card value, usable ace, stick/hit)
        self.action_values = np.zeros(state_action_shape)
        self.returns = np.zeros_like(self.action_values)
        self.policy = np.ones_like(self.action_values)/action_space.n
        self.state_visit_counts = np.ones_like(self.action_values)

        self.actions = [0,1]
        self.epsilon = 0.001
        self.discount = 1.


        if not np.all(self.policy.sum(-1) == 1):
            raise Exception("Probability of actions do not sum up to 1!")
        
    def sample_action(self, state):
        action = np.random.choice(self.actions, p=self.policy[state])
        return int(action)
    
    def check_already_visited(self, first_visit_list, state):
        for first_state in first_visit_list:
            elements_equal = True
            for a, b in zip(first_state, state):
                if a != b:
                    elements_equal = False
            if elements_equal == True:
                return True
        return False
    
    def get_state_values(self, state):
        return sum([self.action_values[state+(action,)]*self.policy[state+(action,)] for action in self.actions])
    
    def compute_return(self, episode, t):
        discount = 1
        g_return = 0
        for _, _, reward_t_T in episode[t:]:
            g_return += discount*reward_t_T
            discount *= self.discount

        return g_return
    
    def update_policy(self, episode):   # episode -> [(s,a,r), (s',a',r'), (s",a",r"), ...]
        # update state action values from episode
        first_visit_list = []
        for t, sar in enumerate(episode):
            state = sar[0]
            action = sar[1]
            state_action = state + (action,)
            if not self.check_already_visited(first_visit_list, state_action):  # might not work, depends if tuple can be compared
                first_visit_list.append(state_action)

                g_return = self.compute_return(episode, t)

                self.returns[state_action] = self.returns[state_action] + (1/self.state_visit_counts[state_action])*(g_return - self.returns[state_action])
                self.state_visit_counts[state_action] += 1
                self.action_values[state_action] = self.returns[state_action]

        # update policy according to new state action values
        for state, _, _ in episode:
            a_star = self.actions[np.argmax(self.action_values[state])]

            for a in self.actions:
                if a == a_star:
                    self.policy[state][a] = 1-self.epsilon+(self.epsilon/len(self.actions))
                else:
                    self.policy[state][a] = self.epsilon/len(self.actions)
        

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    env = gym.make('Blackjack-v1', natural=False, sab=True)

    agent = OnPolicyFirstVistMCSoft(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    num_episodes = 1000000

    episodes = []
    episode_return_list = []

    for e in tqdm(range(num_episodes)):
        observation, info = env.reset()
        terminated = False
        truncated = False
        single_episode = []
        while not terminated and not truncated:
            #action = env.action_space.sample()  # agent policy that uses the observation and info
            action = agent.sample_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            single_episode.append((observation, action, reward))
        
        agent.update_policy(single_episode)

        episode_return = agent.compute_return(single_episode, t=0)

        episodes.append(single_episode)
        episode_return_list.append(episode_return)
    

    env.close()

    save_object(agent, f"blackjack_MC_soft_ep_{num_episodes}.pkl")
        