# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python

# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------

def policy_evaluation(policy, env, gamma, theta):
    V = np.zeros(n_states) # Initialize value function V(s) to 0 for all states
    while True:
        delta = 0
        for s in range(n_states):
            v = 0
            # For each action according to the current policy
            for a, action_prob in enumerate(policy[s]):
                # Sum over possible next states (s'), rewards (r), and probabilities (p)
                for prob_s_prime, s_prime, reward, _ in env.P[s][a]:
                    v += action_prob * prob_s_prime * (reward + gamma * V[s_prime])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V



# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------

def policy_improvement(env, V, gamma):
    policy = np.zeros([n_states, n_actions]) / n_actions # Initialize policy to be equiprobable random policy
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob_s_prime, s_prime, reward, _ in env.P[s][a]:
                action_values[a] += prob_s_prime * (reward + gamma * V[s_prime])
        best_action = np.argmax(action_values)
        policy[s] = np.eye(n_actions)[best_action] # Update policy to be greedy
    return policy

#-------------------------------------------------
# Policy Iteration
# -------------------------------------------------

def policy_iteration(env, gamma, theta):
    policy = np.zeros([n_states, n_actions]) # Start with an arbitrary policy (e.g., all zeros, which will be updated)
    # Initialize policy to be equiprobable random policy
    for s in range(n_states):
        policy[s] = np.ones(n_actions) / n_actions

    # Display initial policy and value function
    print_policy(policy, title="Initial Policy:")
    V_initial = policy_evaluation(policy, env, gamma, theta)
    print_value_function(V_initial, title="Initial Value Function (after first evaluation):")


    while True:
        V = policy_evaluation(policy, env, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, V

```

## Output


<img width="1243" height="1266" alt="image" src="https://github.com/user-attachments/assets/04486217-b652-4a79-9cf8-7e9722009c86" />


## Result

The Policy Iteration algorithm was successfully implemented on the FrozenLake-v1 (4×4) environment using Gymnasium. The algorithm converged after 3 policy iterations, producing the optimal state-value function and the optimal policy for navigating the environment.


## Inference

The experiment demonstrates that Policy Iteration efficiently solves a Markov Decision Process by alternately performing policy evaluation and policy improvement until the policy becomes stable. The obtained optimal policy guides the agent to maximize the expected cumulative reward while avoiding hole states and reaching the goal. The resulting state-value function indicates the expected return from each state under the optimal policy.

