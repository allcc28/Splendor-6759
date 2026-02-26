# Copilot Instructions for Splendor RL Project

## Project Overview
IFT6759 course project: Training RL agents for the board game Splendor using different reward shaping approaches (Score-based → Event-based → AlphaZero-style planning). Current phase: **PPO Score-based agent trained and validated**.

## Critical Architecture Patterns

### 1. Hybrid Directory Structure
```
modules/           # Legacy codebase (gym_splendor_code, agents, arena)
project/           # New development (PPO agents, training, experiments)
legacy/            # Archived old experiments
docs/              # Project documentation & plans
```

**Key insight**: The codebase maintains backward compatibility with legacy code while building new components in `project/`. Always check both locations when searching for implementations.

### 2. Python Path Magic (`sitecustomize.py`)
```python
# Auto-adds modules/ to PYTHONPATH at Python startup
MODULES_DIR = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(MODULES_DIR))
```

**Import conventions**:
- Legacy Splendor env: `from gym_splendor_code.envs.splendor import SplendorEnv`
- New utilities: Use **relative imports** within `project/src/`: `from .state_vectorizer import SplendorStateVectorizer`
- Never use absolute `from utils.X` inside `project/src/utils/` (causes ModuleNotFoundError)

### 3. State Representation (135-dim Fixed Vector)
Splendor has variable-length observations (cards/nobles change). The **vectorizer** converts to fixed-size input:

```python
# project/src/utils/state_vectorizer.py
vector = [
    active_player_hand (35),   # Gems, discounts, VP, reserved cards
    opponent_hand (14),         # Simplified opponent view
    board_gems (6),             # Available gem tokens
    board_cards (72),           # 12 cards x 6 features (one-hot row/discount + VP/price)
    board_nobles (6),           # 3 nobles x 2 features
    game_progress (2)           # Turn count, active player flag
]  # Total: 135 dims, normalized to [0, 1], dtype=float32
```

**Design spec**: [project/docs/development/specs/state_representation_spec.md](project/docs/development/specs/state_representation_spec.md)

### 4. Gym Wrapper for SB3
```python
# project/src/utils/splendor_gym_wrapper.py
class SplendorGymWrapper(gym.Env):
    observation_space = Box(0.0, 1.0, (135,), float32)
    action_space = Discrete(200)  # Max possible legal actions
    
    def step(self, action_idx: int):
        # Maps action index to actual Splendor Action object
        action = self.cached_legal_actions[action_idx]
```

**Critical**: Actions are **indices into `cached_legal_actions` list**, not Action objects directly. Invalid action_idx (>= len) returns -10 reward and terminates.

**Reward modes** (configurable):
- `score_only`: Pure score difference (sparse)
- `score_progress`: +0.01/action + score_diff + 50*win (used in current training)
- `event_based`: Custom event rewards (planned Phase 2)

### 5. Training Pipeline (PPO)
```bash
# Full 1M timestep training
python project/scripts/train_score_based.py

# Uses config from project/configs/training/ppo_score_based.yaml
# Outputs to: project/logs/ppo_score_based_v1_YYYYMMDD_HHMMSS/
#   ├── final_model.zip          # SB3 model file
#   ├── logs/tensorboard/         # Training curves
#   ├── logs/checkpoints/         # Every 50K steps
#   └── config.yaml               # Training config snapshot
```

**Monitoring**: `tensorboard --logdir project/logs`

## Development Workflows

### Environment: WSL2 + GPU
**Critical**: All training/testing runs in WSL2, not Windows PowerShell.

```bash
# Activate environment
wsl
cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759
conda activate splendor

# Run training (background)
tmux new-session -d -s train 'python project/scripts/train_score_based.py 2>&1 | tee training.log'
tmux attach -t train  # Ctrl+B, D to detach

# Check GPU
nvidia-smi
```

**Hardware**: RTX 4090 (24GB), Threadripper 5955WX (32 threads)

### Testing
```bash
# Run all tests
pytest project/tests/ -v

# Test specific module
pytest project/tests/test_state_vectorizer.py -v
pytest project/tests/test_gym_wrapper.py -v
```

**Test pattern**: Every vectorizer/wrapper component has comprehensive tests (13+ tests for vectorizer, 11+ for wrapper). Tests validate:
- Shape/dtype correctness
- Determinism (same state → same vector)
- State change detection (different states → different vectors)
- Normalization ([0, 1] range)
- SB3 compatibility (`check_env()`)

### Documentation Standards
```
project/docs/development/
├── PROGRESS.md                    # Task tracker (update after each phase)
├── dev_logs/YYYY-MM-DD_*.md      # Session logs
├── specs/*.md                     # Technical specifications
└── architecture/*.md              # Design decisions (ADRs)
```

Update `PROGRESS.md` when completing tasks. Use ADRs for architectural decisions (e.g., ADR-001: Why PPO over DQN).

## Project-Specific Conventions

### 1. Splendor Game API
```python
# Legacy environment interface (DO NOT modify - used by all agents)
env = SplendorEnv(opponent=RandomAgent())
obs, reward, done, info = env.step('deterministic', action)

# State structure
state = env.current_state_of_the_game  # State object
gems = state.board.gems_on_board[GemColor.RED]
cards = state.board.cards_on_board[Row.CHEAP]
player = state.list_of_players_hands[player_id]
```

### 2. Config Management
All experiments use YAML configs in `project/configs/`:
```yaml
# Training config structure
training:
  total_timesteps: 1000000
  algorithm: ppo
  network: [256, 256, 128]  # MLP layers

environment:
  env_name: splendor
  reward_mode: score_progress
  
ppo:
  learning_rate: 0.0003
  batch_size: 64
  n_steps: 2048
```

Load via: `yaml.safe_load(open('project/configs/training/ppo_score_based.yaml'))`

### 3. Opponent Configuration
```python
# Always specify opponent when creating Splendor env
from modules.agents.random_agent import RandomAgent

env = SplendorEnv(opponent=RandomAgent())  # For training
env = SplendorEnv(opponent=GreedyAgent())  # For evaluation
```

Opponents: `RandomAgent`, `GreedyAgent`, `ValueNNAgent`, `MCTSAgent` (in `modules/agents/`)

### 4. File Naming Conventions
- Scripts: `train_*.py`, `evaluate_*.py`, `test_*.py`
- Configs: `ppo_*.yaml`, `mcts_*.yaml`, `reward_*.yaml`
- Logs: `ppo_score_based_v1_20260224_112345/` (versioned + timestamped)
- Tests: `test_*.py` (pytest discovery)

## Current Implementation Status

**Completed (Phase 1-5)**:
- ✅ WSL2 + GPU environment (PyTorch 2.5.1 + CUDA 12.1, SB3 2.7.1)
- ✅ 135-dim state vectorizer (`project/src/utils/state_vectorizer.py`)
- ✅ SB3 Gym wrapper (`project/src/utils/splendor_gym_wrapper.py`)
- ✅ PPO training script (`project/scripts/train_score_based.py`)
- ✅ 1M timestep training completed (reward: -9.91 → +27.99)

**In Progress**:
- Phase 6: Evaluation vs RandomAgent (win rate measurement)

**Planned** (see [docs/plan.md](docs/plan.md)):
- Phase 2: Event-based reward shaping (Weeks 4-7)
- Phase 3: AlphaZero-style MCTS agent (Weeks 8-10)

## Common Pitfalls

1. **Import errors in `project/src/utils/`**: Always use relative imports (`.state_vectorizer`), never absolute (`utils.state_vectorizer`)
2. **Action type mismatch**: Gym wrapper expects action **index** (int), not Action object
3. **GPU out of memory**: Current training uses ~1.4GB VRAM (5.6% of RTX 4090). If OOM occurs, reduce batch_size or network size
4. **Windows vs WSL paths**: Training runs in WSL (`/mnt/c/...`), but VS Code uses Windows paths (`C:\...`). Use WSL for all Python execution
5. **Stale tmux sessions**: Always check `tmux ls` before starting new training to avoid duplicate runs

## Quick Reference

**Key Files**:
- State vectorizer: [project/src/utils/state_vectorizer.py](project/src/utils/state_vectorizer.py)
- Gym wrapper: [project/src/utils/splendor_gym_wrapper.py](project/src/utils/splendor_gym_wrapper.py)
- Training script: [project/scripts/train_score_based.py](project/scripts/train_score_based.py)
- Progress tracker: [project/docs/development/PROGRESS.md](project/docs/development/PROGRESS.md)
- Project plan: [docs/plan.md](docs/plan.md)

**Example Training Run**:
```bash
# Start training
wsl -e bash -c "cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759 && conda activate splendor && python project/scripts/train_score_based.py"

# Monitor
wsl -e bash -c "cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759 && tail -f training.log"
```

**Load Trained Model**:
```python
from stable_baselines3 import PPO
model = PPO.load("project/logs/ppo_score_based_v1_20260224_113524/final_model")
obs, info = env.reset()
action, _states = model.predict(obs, deterministic=True)
```
