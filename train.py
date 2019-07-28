import torch
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


            #while True:                                 # Consider the agent will solve the environment
            for t in range(1, self.n_timesteps):
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, done):
                    agent.step(state, action, reward, next_state, done)

            scores_window.append(np.max(scores))
            scores_agents.append(np.max(scores))
            # Checkpoint
            self.checkpoint(i_episode, scores_window,
                            exp_number,
                            self.display_freq,
                            self.save_at_checkpoint)

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
