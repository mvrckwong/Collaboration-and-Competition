# Project Report

## 1. Multi-Agent Deep Deterministic Policy Gradient

Multi-Agent DDPG trains two agents in parallel. Based on the rewarding system of the environment, the agent collaborates and competes. "Just doing a simple extension of single agent RL by independently training the two agents does not work very well because the agents are independently updating their policies as learning progresses. And this causes the environment to appear non-stationary from the viewpoint of any one agent."

## 2. Summary of Params and Hyperparams of the Agent of the Network

*Network Hyperparameters:*
```
tau (interpolation parameter soft update): 0.001
lr_actor (learning rate actor): 1e-3
lr_critic (learning rate critic): 1e-4
```
*Agent Parameters / Hyperparameters:*
```
Gamma or Discount Rate: 0.99
Buffer Size: 1e5
Batch Size: 100
Update Every: 1
```
*Training Parameters:*
```
Total Number of Episodes: 200
Maximum Number of Timesteps per Episodes: 2000
```

## 3. Final Results and Takeaways:

## 4. Further Improvements / Further Works

I would try to train the agents in MAPPO or Multi-Agent Proximal Policy Optimization. I would want to compare the results with MADDPG.
