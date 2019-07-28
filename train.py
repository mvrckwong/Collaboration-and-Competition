from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from collections import deque


### Initialize the Agent
class initialize:
    def __init__(self,
                 env=None,
                 name="MADDPG",
                 n_episodes=50,
                 n_timesteps=1000,
                 n_steps=2):

        self.env = env
        self.name = name

        self.state_size = None
        self.action_size = None
        self.n_agents = None
        self.agent = Agent(self.state_size, self.action_size, random_seed)

        self.n_episodes = n_episodes
        self.n_timesteps = n_timesteps
        self.n_steps = n_steps

        self.display_freq = 100
        self.save_at_checkpoint = True

    def train(self, exp_number=0):
        scores = []
        scores_window = deque(maxlen=self.display_freq)

        for i_episode in range(1, self.n_episodes+1):
            env_info = self.env.reset(train_mode=True)[brain_name]       # Reset the environment
            states = env_info.vector_observations                   # Get the current state
            self.agent.reset()
            scores = np.zeros(num_agents)

            #while True:                                 # Consider the agent will solve the environment
            for t in range(1, self.n_timesteps):
                actions = self.agent.act(states)
                env_info = self.env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, done):
                    self.agent.step(state, action, reward, next_state, done)

            scores_window.append(np.max(scores))
            scores_agents.append(np.max(scores))
            # Checkpoint
            self.checkpoint(i_episode, scores_window,
                            exp_number,
                            self.display_freq,
                            self.save_at_checkpoint)

            if np.mean(scores_window) >= 0.5 and i_episode >= 100:
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                print('\rEnvironment solved in {} episodes (average score {:.2f}).'.format(i_episode-100,np.mean(scores_deque)))
                break

        return scores_agent

    def checkpoint(self, i_episode,
                   scores_window,
                   exp_number,
                   display_freq=100,
                   save_at_checkpoint=False):

        print("\rEpisode {}\t\tAvg Score: {:.2f}".format(i_episode, np.mean(scores_window), end=""))
        if i_episode % display_freq == 0:
            print("\rEpisode {}\t\tAvg Score: {:2f}".format(i_episode, np.mean(scores_window)))

        # Save Model for every 100 episodes
        if i_episode == 100:
            #torch.save(policy.state_dict(), model_path + "{2}_{1}_Chkpnt{0}.pth".format(i_episode, exp_number, model_name))

    def test(self, play_episodes):
        self.agent.actor_local.load_state_dict(torch.load("checkpoint_actor.pth"))
        self.agent.critic_local.load_state_dict(torch.load("checkpoint_critic.pth"))

        for i in range(1, play_episodes+1):                             # play game for number of episodes
            env_info = env.reset(train_mode=False)[brain_name]          # reset the environment
            states = env_info.vector_observations                       # get the current state (for each agent)
            scores = np.zeros(num_agents)                               # initialize the score (for each agent)

            while True:
                actions = agent.act(states)                             # select an action (for each agent)
                actions = np.clip(actions, -1, 1)                       # all actions between -1 and 1
                env_info = env.step(actions)[brain_name]                # send all actions to the environment
                next_states = env_info.vector_observations              # get next state (for each agent)
                rewards = env_info.rewards                              # get reward (for each agent)
                dones = env_info.local_done                             # see if episode finished
                scores += env_info.rewards                              # update the score (for each agent)
                states = next_states                                    # roll over states to next time step
                if np.any(dones):                                       # exit loop if episode finished
                    break
            print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
