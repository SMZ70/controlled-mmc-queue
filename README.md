<h1 align="center">Reinforcement Learning for Controlled M/M/c Queues</h1>



## About The Project

An implementation of different reinforcement learning algorithms for controlled M/M/c queues. Some of the algorithms, e.g., A2C are implemented using [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

### Prerequisites

* stable-baselines3

## Usage
- For example to use tabular queue learning for a system with $\lambda = 2$ and $\mu_1=2,$ $\mu_2=0.1$

```python
from controlledmmcqueue import TabularQLearner

# create the learner instance
learner = TabularQLearner(mmc_env=env,
                          learning_rate=0.25, discount_factor=0.95,  
                          max_eps=1, min_eps=0.01, eps_decay=0.01)

# run the algorithm to learn the optimal Q table
rewards, epsilons = tab_q_learner.learn(500, 
                                        log_progress=True, 
                                        return_history=True)
```

- To run a simple simulation on the system using the learned policy:

```python
learner.simulate(n_events=n_events,
                 n_reps=n_reps,
                 verbose=2,
                 init_state=[0,0,0])

```
_For more examples,  and an illustrative comparison of the methods please refer to the [Comparison Jupyter Notebook](comparison.ipynb)_

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

