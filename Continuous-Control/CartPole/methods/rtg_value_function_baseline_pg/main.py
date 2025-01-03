import gymnasium as gym
from torch import nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import wandb
import os
import argparse
from distutils.util import strtobool
from gymnasium.wrappers import RecordVideo


PATH = os.path.dirname(os.path.abspath(__file__))
NUM_EPOCHS = 80
SEED = 42
BATCH_SIZE = 5000
LEARNING_RATE = 1e-2
PATH_POLICY_MODEL = PATH + "/training_statistics/policy_model_cpu_2.pth"
PATH_VALUE_MODEL = PATH + "/training_statistics/value_model_cpu_2.pth"
PATH_METRICS = PATH + "/training_statistics/metrics_dict_cpu_2.pkl"


class MLP(nn.Module):
    def __init__(self, layer_sizes: list, activation_func: nn.modules.activation, output_activation_func: nn.modules.activation):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(activation_func())
            else:
                layers.append(output_activation_func())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_policy(policy_model: nn.Module, observation: torch.Tensor) -> torch.distributions:
    logits = policy_model(observation)
    action_probs = torch.distributions.Categorical(logits=logits)
    return action_probs

def get_action(policy_model: nn.Module, observation: torch.Tensor) -> torch.Tensor:
    return get_policy(policy_model, observation).sample().item()

def compute_loss_value_model(value_estimates: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    return torch.square(value_estimates-rewards).mean()

def compute_loss_policy_model(policy_model: nn.Module, value_estimates: nn.Module, observation: torch.Tensor, action: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    log_prob = get_policy(policy_model, observation).log_prob(action)
    return -torch.matmul(log_prob, rewards) / BATCH_SIZE
    #return -(log_prob * (rewards-value_estimates)).mean()

def reward_to_go(trajectory_rewards: list) -> list:
    rtg = []
    for i in range(len(trajectory_rewards)):
        rtg.append(sum(trajectory_rewards[i:]))
    return rtg


def train(log: bool, device: str):

    if device == "cuda":
        "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    env = gym.make("CartPole-v1")

    policy_model = MLP(
        layer_sizes=[env.observation_space.shape[0], 32, env.action_space.n.item()],
        activation_func=nn.Tanh,
        output_activation_func=nn.Identity
    ).to(device)
    policy_optim = torch.optim.Adam(policy_model.parameters(), lr=LEARNING_RATE)

    value_model = MLP(
        layer_sizes=[env.observation_space.shape[0], 32, 1],
        activation_func=nn.Tanh,
        output_activation_func=nn.Identity
    ).to(device)
    value_optim = torch.optim.Adam(value_model.parameters(), lr=LEARNING_RATE)


    observation, info = env.reset(seed=SEED)

    epoch_loss_policy = []
    epoch_loss_value = []
    mean_epoch_reward = []
    mean_epoch_len = []

    
    for _ in tqdm(range(NUM_EPOCHS)):

        batch_rewards_rtg = []
        batch_rewards = []
        batch_obs = []
        #time_step_list = []
        batch_actions = []
        batch_lens = []

        for _ in range(BATCH_SIZE):
            done = False
            trajectory_rewards = []
            trajectory_obs = []
            trajectory_actions = []
            #t = 0
            while not done:
                trajectory_obs.append(observation)
                action = get_action(policy_model, torch.as_tensor(observation, dtype=torch.float32).to(device))
                observation, reward, terminated, truncated, info = env.step(action)

                trajectory_actions.append(action)
                trajectory_rewards.append(reward)

                if terminated or truncated:
                    done = True

                #time_step_list.append(t)
                #t += 1

            batch_lens.append(len(trajectory_rewards))
            batch_rewards_rtg += reward_to_go(trajectory_rewards)
            batch_rewards += [sum(trajectory_rewards)] * len(trajectory_rewards)
            batch_obs += trajectory_obs
            batch_actions += trajectory_actions

            observation, info = env.reset()
        
        #time_step_oh = torch.nn.functional.one_hot(torch.tensor(time_step_list), num_classes=500)
        #value_model_input = torch.cat((torch.as_tensor(np.array(batch_obs), dtype=torch.float32), time_step_oh), dim=1)
        value_model_input = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)

        value_estimates = value_model(value_model_input.to(device)).squeeze()
        batch_loss_value_model = compute_loss_value_model(value_estimates, torch.as_tensor(np.array(batch_rewards_rtg), dtype=torch.float32).to(device))
                                                          
        batch_loss = compute_loss_policy_model(
            policy_model,
            value_estimates.detach(),
            torch.as_tensor(np.array(batch_obs), dtype=torch.float32).to(device),
            torch.as_tensor(np.array(batch_actions), dtype=torch.int32).to(device),
            torch.as_tensor(np.array(batch_rewards_rtg), dtype=torch.float32).to(device)
        )

        batch_loss.backward()
        policy_optim.step()
        policy_optim.zero_grad()

        batch_loss_value_model.backward()
        value_optim.step()
        value_optim.zero_grad()

        ## gather eval metics ##
        epoch_loss_policy.append(batch_loss.detach().cpu().numpy())
        epoch_loss_value.append(batch_loss_value_model.detach().cpu().numpy())
        mean_epoch_reward.append(np.mean(batch_rewards))
        mean_epoch_len.append(np.mean(batch_lens))

        # log metrics to wandb
        if log == True:
            wandb.log(
                {
                    "epoch_loss_policy": epoch_loss_policy[-1], 
                    "epoch_loss_value": epoch_loss_value[-1],
                    "mean_epoch_reward": mean_epoch_reward[-1],
                    "mean_epoch_length": mean_epoch_len[-1]
                }
            )


    metrics_dict = {
        "epoch_loss_policy": epoch_loss_policy, 
        "epoch_loss_value": epoch_loss_value,
        "mean_epoch_reward": mean_epoch_reward,
        "mean_epoch_len": mean_epoch_len
    }
    
    ## save policy_model and metrics ##
    with open(PATH_METRICS, 'wb') as f:
        pickle.dump(metrics_dict, f)

    torch.save(policy_model.state_dict(), PATH_POLICY_MODEL)
    torch.save(value_model.state_dict(), PATH_VALUE_MODEL)

    env.close()
    if log == True:
        wandb.finish()

def evaluate(device: str):
    env = gym.make("CartPole-v1", render_mode="human")

    policy_model = MLP(
        layer_sizes=[env.observation_space.shape[0], 32, env.action_space.n.item()],
        activation_func=nn.Tanh,
        output_activation_func=nn.Identity
    ).to(device)

    policy_model.load_state_dict(torch.load(PATH_POLICY_MODEL, weights_only=True))
    policy_model.eval()

    with open(PATH_METRICS, 'rb') as f:
        metrics_dict = pickle.load(f)

    plot_metrics(**metrics_dict)

    observation, info = env.reset(seed=SEED)
    done = False
    while not done:
        action = get_action(policy_model, torch.as_tensor(observation, dtype=torch.float32).to(device))
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            done = True

    record_agent(policy_model, device) 


def plot_metrics(epoch_loss_policy, epoch_loss_value, mean_epoch_reward, mean_epoch_len):
    plt.plot(epoch_loss_policy, label="epoch loss policy")
    plt.plot(mean_epoch_reward, label="mean reward")
    plt.plot(mean_epoch_len, label="mean trajectory length")

    plt.xlabel("# Epochs")

    plt.legend()
    plt.savefig(PATH+"/plots"+"/training_stats_2_1.pdf")
    plt.savefig(PATH+"/plots"+"/training_stats_2_1.jpg")
    plt.show()
    
    # plot in separate figure because of different magnitude on y-axis
    plt.plot(epoch_loss_value, label="epoch loss value")
    plt.xlabel("# Epochs")

    plt.legend()
    plt.savefig(PATH+"/plots"+"/training_stats_2_2.pdf")
    plt.savefig(PATH+"/plots"+"/training_stats_2_2.jpg")
    plt.show()

def record_agent(model: nn.Module, device: str):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=PATH+"/videos", name_prefix="eval")

    observation, info = env.reset(seed=SEED)
    done = False
    while not done:
        action = get_action(model, torch.as_tensor(observation, dtype=torch.float32).to(device))
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            done = True
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Simple Policy Gradient Method.',
        description='Solving the CartPole problem using vanilla policy gradient.'
    )

    parser.add_argument('-t', '--task', type=str, default="eval")
    parser.add_argument('-l', '--log', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('-d', '--device', type=str, default="cpu")
    parser.add_argument('-n', '--name', type=str, default="nameless_run")
    args = parser.parse_args()
    print(args)

    if args.log == True:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="CartPole",
            name=args.name,
            # track hyperparameters and run metadata
            config={
            "learning_rate": LEARNING_RATE,
            "architecture": "MLP",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE
            }
        )
    
    if args.task == "train":
        train(args.log, args.device)
    elif args.task == "eval":
        evaluate(args.device)    
