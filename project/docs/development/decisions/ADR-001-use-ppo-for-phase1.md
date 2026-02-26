# ADR-001: Use PPO for Phase 1 RL Agents

**Status**: Accepted  
**Date**: 2026-02-24  
**Deciders**: Yehao Yan, Team  
**Technical Story**: US-1.2 (Score-based Agent), US-1.3 (Event-based Agent)

---

## Context

We need to choose a reinforcement learning algorithm for Phase 1 agents (Score-based and Event-based). The algorithm must:

1. Handle sparse rewards (Score-based agent challenge)
2. Work with discrete action spaces (Splendor has ~50-200 legal actions per turn)
3. Be stable enough for 10k+ episode training
4. Have good library support for fast implementation
5. Be explainable in academic paper/presentation

**Candidates Considered**:
- Proximal Policy Optimization (PPO)
- Deep Q-Network (DQN)
- Actor-Critic (A2C/A3C)
- Soft Actor-Critic (SAC)

---

## Decision

**We will use Proximal Policy Optimization (PPO)** for both Score-based and Event-based agents in Phase 1.

Implementation:
- Use **Stable-Baselines3** library (mature, well-documented)
- Standard PPO hyperparameters as starting point
- Shared network architecture for fair comparison between score/event agents

---

## Alternatives Considered

### Alternative 1: Deep Q-Network (DQN)

**Description**: Value-based method that learns Q(s,a) to select actions.

**Pros**:
- Simpler conceptual model (just learn action values)
- Off-policy learning = more sample efficient
- Experience replay buffer = better data utilization
- Well-established for game AI (original Atari paper)

**Cons**:
- Struggles more with large/varying action spaces (Splendor: 50-200 actions)
- Can be unstable with function approximation
- Target network updates can lag behind current policy
- Overestimation bias issues

**Why Not Chosen**: 
Splendor's dynamic action space (legal moves change each turn) is awkward for Q-learning. Would need action masking in every step, adding complexity. PPO's policy-based approach handles this more naturally.

---

### Alternative 2: A2C/A3C (Advantage Actor-Critic)

**Description**: Combines policy gradient (actor) with value function (critic).

**Pros**:
- Actor-critic architecture (like PPO but simpler)
- Lower variance than pure policy gradient
- Asynchronous version (A3C) exploits our 32-thread CPU

**Cons**:
- Less stable than PPO (no clipped updates)
- A3C hard to reproduce (synchronization issues)
- A2C often outperformed by PPO in practice

**Why Not Chosen**:
PPO is essentially "A2C done right" - it adds trust region constraint via clipping, making training much more stable. No reason to use the older, less stable version.

---

### Alternative 3: Soft Actor-Critic (SAC)

**Description**: Off-policy actor-critic with entropy regularization.

**Pros**:
- State-of-the-art for continuous control
- Sample efficient (off-policy)
- Entropy bonus encourages exploration

**Cons**:
- Designed for continuous action spaces
- More complex (twin critics, temperature tuning)
- Overkill for our discrete action problem

**Why Not Chosen**:
SAC is designed for robotics/continuous control. Splendor has discrete actions. Using SAC would require Gumbel-Softmax tricks or discretization hacks. Not worth the complexity.

---

## Consequences

### Positive
- ✅ **Stability**: PPO's clipped objective prevents destructive policy updates
- ✅ **Library Support**: Stable-Baselines3 provides battle-tested PPO implementation
- ✅ **Handles Dynamic Actions**: Policy output is probability distribution - easy to mask invalid actions
- ✅ **Industry Standard**: PPO is the go-to for game AI (OpenAI Five, AlphaStar used PPO-variants)
- ✅ **Hyperparameter Robustness**: PPO less sensitive to learning rate and other hyperparameters
- ✅ **Explainability**: Policy gradient methods are more interpretable than Q-learning for papers

### Negative
- ⚠️ **On-Policy Learning**: Must collect new data for each update (cannot reuse old experiences)
- ⚠️ **Sample Efficiency**: Potentially slower than DQN in wall-clock time per episode
- ⚠️ **Memory Usage**: Stores full rollout buffer (states, actions, rewards) before update

### Neutral
- Multiple epochs over same batch (PPO feature) helps mitigate on-policy inefficiency
- Our 32-thread CPU enables massive parallel self-play to overcome sample requirements

---

## Implementation Notes

### Stable-Baselines3 Configuration

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Standard PPO hyperparameters (will tune if needed)
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=2048,           # Rollout buffer size per env
    batch_size=64,          # Minibatch size for updates
    n_epochs=10,            # Epochs per rollout buffer
    learning_rate=3e-4,     # Standard Adam LR
    clip_range=0.2,         # PPO clipping epsilon
    ent_coef=0.01,          # Entropy bonus for exploration
    vf_coef=0.5,            # Value function loss coefficient
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE parameter
    verbose=1,
    tensorboard_log="./logs/tensorboard/"
)
```

### Multi-Process Self-Play

Utilize Threadripper's 32 threads:
```python
# Create 16 parallel environments (leave some cores for OS)
env = SubprocVecEnv([make_env(i) for i in range(16)])
```

This gives us 16× data collection speed.

---

## Related Decisions

- [ADR-002]: Score-based reward function design (see `score_based_agent_design.md`)
- [ADR-003]: Training monitoring and experiment tracking (TBD)

---

## References

1. Schulman et al. (2017): "Proximal Policy Optimization Algorithms"  
   https://arxiv.org/abs/1707.06347

2. Stable-Baselines3 Documentation:  
   https://stable-baselines3.readthedocs.io/

3. OpenAI Spinning Up - PPO:  
   https://spinningup.openai.com/en/latest/algorithms/ppo.html

4. Huang et al. (2022): "The 37 Implementation Details of PPO"  
   (Shows PPO's robustness across many domains)

---

**Decision Made**: 2026-02-24  
**Review Date**: After initial experiments (if PPO shows problems, can reconsider)
