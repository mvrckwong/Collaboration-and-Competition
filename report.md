# Project Report

## 1. Multi-Agent Deep Deterministic Policy Gradient

MADDPG is a type of Actor-Critic based methods in reinforcement learning. Main difference between other types of RL is in makes use of critic network which takes care of the variance problem. Core implementation is similar to the previous one. Although, in here, two or various agents run in parallel, the agent can either collaborate or compete in solving the environment. "By independently training the two agents does not work very well because the agents are independently updating their policies as learning progresses. And this causes the environment to appear non-stationary from the viewpoint of any one agent."

## 2. Summary of Params and Hyperparams of the Agent of the Network

*Network Hyperparameters:*
```
TAU = 7e-2
LR_ACTOR = 1e-4
LR_CRITIC = 5e-4
```
*Agent Parameters / Hyperparameters:*
```
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 128
GAMMA = 0.99
WEIGHT_DECAY = 0
UPDATE_EVERY = 2
NUM_UPDATES = 20
```
*Training Parameters:*
```
Total Number of Episodes: 25000
Maximum Number of Timesteps per Episodes: 700
```

## 3. Final Results and Takeaways:

I was not able to save the graph. I had error when plotting the graph:
```
Traceback (most recent call last):
  File "train.py", line 100, in <module>
    plot_process(scores)
  File "train.py", line 41, in plot_process
    fig = plt.figure(figsize=(16,5))
NameError: name 'plt' is not defined
```
I did train the network on the local console. Here are the following results:
```
Episode 100             Avg Score: 0.046000     Max Score: 0.09    
Episode 200             Avg Score: 0.095400     Max Score: 0.10    
Episode 300             Avg Score: 0.089300     Max Score: 0.10    
Episode 400             Avg Score: 0.113100     Max Score: 0.20    
Episode 500             Avg Score: 0.137200     Max Score: 0.10    
Episode 600             Avg Score: 0.162100     Max Score: 0.10    
Episode 700             Avg Score: 0.133900     Max Score: 0.10    
Episode 800             Avg Score: 0.143900     Max Score: 0.10    
Episode 900             Avg Score: 0.125200     Max Score: 0.10    
Episode 1000            Avg Score: 0.168000     Max Score: 0.20    
Episode 1100            Avg Score: 0.196700     Max Score: 0.10    
Episode 1200            Avg Score: 0.206900     Max Score: 0Episode
Episode 1300            Avg Score: 0.234800     Max Score: 0.60: 0.
Episode 1400            Avg Score: 0.197300     Max Score: 0.19    
Episode 1500            Avg Score: 0.290200     Max Score: 0.19    
Episode 1600            Avg Score: 0.335600     Max Score: 0.10    
Episode 1700            Avg Score: 0.250400     Max Score: 0.10    
Episode 1800            Avg Score: 0.246800     Max Score: 0.20    
Episode 1900            Avg Score: 0.367800     Max Score: 0.40    
Episode 2000            Avg Score: 0.184400     Max Score: 0.60    
Episode 2100            Avg Score: 0.134800     Max Score: 0.20    
Episode 2200            Avg Score: 0.247400     Max Score: 0.30    
Episode 2300            Avg Score: 0.187500     Max Score: 0.10    
Episode 2400            Avg Score: 0.132000     Max Score: 0.10    
Episode 2500            Avg Score: 0.144700     Max Score: 0.10    
Episode 2600            Avg Score: 0.133300     Max Score: 0.10    
Episode 2700            Avg Score: 0.165500     Max Score: 0.30    
Episode 2800            Avg Score: 0.115900     Max Score: 0.10    
Episode 2900            Avg Score: 0.167200     Max Score: 0.20    
Episode 3000            Avg Score: 0.124200     Max Score: 0.10    
Episode 3100            Avg Score: 0.277300     Max Score: 1.90    
Environment solved in 3124 episodes (average score 0.50).: 1.80    
```

The environment has been solved around 3124 number of episodes. The Batch size, buffer size and tau learning rate was fine tuned in order to achieved the desired results in a faster episodes.

## 4. Further Improvements / Further Works

- Multi-Agent Proximal Policy Gradient
- Agent learns from raw pixel data. (Convolutional Neural Network).
- Continue Learning in different model agent.
- Tuning the network further.
