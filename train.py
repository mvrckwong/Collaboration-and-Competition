from unityagents import UnityEnvironment
import numpy as np

from agent import Agent
from collections import deque

import torch

# Environment !!!
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]


agent = Agent(state_size, action_size, random_seed=5)

def checkpoint(i_episode, scores_window, scores):
    print("\rEpisode {}\t\tAvg Score: {:.2f}\t\tMax Score: {:2f}".format(i_episode, np.mean(scores_window), np.max(scores)), end="")
    if i_episode % 100 == 0:
        print("\rEpisode {}\t\tAvg Score: {:2f}".format(i_episode, np.mean(scores_window), np.mean(scores)))
    if i_episode % 200 == 0:
        torch.save(agent.actor_local.state_dict(), 'checkpoint_{}_actor.pth'.format(i_episode))
        torch.save(agent.critic_local.state_dict(), 'checkpoint_{}_critic.pth'.format(i_episode))

def plot_process(scores):
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)

    plt.grid(which="major", alpha=0.30)
    plt.title('MADDPG')
    plt.ylabel('Avg Score across all Agents')
    plt.xlabel('Number of Episode')
    plt.savefig('Scores.png')
    plt.legend(loc=0)
    plt.show()


# Train MADDPG
def train(n_episodes=40000,
          n_timesteps=500):
    scores_window = deque(maxlen=100)
    scores_agents = []

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset environment
        states = env_info.vector_observations             # get the current state
        agent.reset()
        scores = np.zeros(num_agents)

        #while True:
        for t in range(1, n_timesteps+1):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]     # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                  # see if episode has finished
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)

            states = next_states
            scores += rewards

            if np.any(dones):
                break

        # from the two scores, we append the better score
        scores_window.append(np.max(scores))
        scores_agents.append(np.max(scores))

        checkpoint(i_episode, scores_window, scores)

        #environment solved for avg score of .5 over past 100 episodes
        if np.mean(scores_window) >= 0.5:
            print('\rEnvironment solved in {} episodes (average score {:.2f}).'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'final_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'final_checkpoint_critic.pth')
            break
    return scores_agents



if __name__ == '__main__':
    scores = train(n_episodes=25000, n_timesteps=700)
    plot_process(scores)
