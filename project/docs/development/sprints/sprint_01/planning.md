# Sprint 1: Reward Shaping Baseline (Phase 1)

**Sprint Duration**: 2026-02-24 to 2026-03-10 (2 weeks)  
**Sprint Goal**: Establish baseline comparison between Score-based and Event-based reward approaches  
**Team**: Yehao Yan, Qianyun Shen, Xinyan Guo

---

## Sprint Objectives

1. ✅ Set up project structure and development workflow
2. ⬜ Implement Score-based RL agent (DQN or PPO)
3. ⬜ Implement Event-based RL agent with dense rewards
4. ⬜ Create training pipeline with configuration management
5. ⬜ Run comparative experiments (500+ games)
6. ⬜ Generate performance analysis and visualization

---

## User Stories

### US-1.1: Project Setup
**As a** developer  
**I want** a clean project structure with proper documentation  
**So that** the team can collaborate efficiently and track progress

**Tasks**:
- [x] Review existing codebase and modules
- [x] Create development documentation structure
- [ ] Set up experiment logging framework
- [ ] Configure GPU environment and dependencies

**Acceptance Criteria**:
- Project structure follows plan.md organization
- Dev logs and ADR templates are ready
- Dependencies installed and verified

---

### US-1.2: Score-Based Agent Implementation
**As a** researcher  
**I want** a baseline RL agent using only game score as reward  
**So that** I can demonstrate the sparse reward problem

**Tasks**:
- [ ] Design agent architecture (DQN vs PPO decision)
- [ ] Implement state representation for neural network input
- [ ] Implement score-based reward function
- [ ] Create training loop with self-play
- [ ] Add checkpointing and model saving

**Acceptance Criteria**:
- Agent can complete full games against random opponent
- Training converges (loss decreases over time)
- Model can be saved/loaded for evaluation
- Win rate against RandomAgent reaches >60% after training

---

### US-1.3: Event-Based Agent Implementation
**As a** researcher  
**I want** an RL agent using dense event-based rewards  
**So that** I can show improved learning in the engine-building phase

**Tasks**:
- [ ] Design event reward function (buy card, reserve, collect gems, noble)
- [ ] Weight tuning for different event types
- [ ] Implement augmented reward calculation
- [ ] Same architecture as Score-based for fair comparison
- [ ] Create training loop

**Acceptance Criteria**:
- Agent receives non-zero rewards during engine-building phase
- Training converges faster than Score-based agent
- Model can be saved/loaded for evaluation
- Win rate against RandomAgent reaches >70% after training

---

### US-1.4: Training Infrastructure
**As a** researcher  
**I want** a flexible training pipeline with configuration management  
**So that** I can easily experiment with different hyperparameters

**Tasks**:
- [ ] Create YAML config system for experiments
- [ ] Implement training script with CLI arguments
- [ ] Add TensorBoard or Weights & Biases logging
- [ ] Implement multi-process self-play (utilize 32 threads)
- [ ] Create utility for resuming training from checkpoint

**Acceptance Criteria**:
- Training can be started with single command
- All hyperparameters configurable via YAML
- Training metrics logged and visualizable
- GPU utilization >80% during training

---

### US-1.5: Evaluation & Comparison
**As a** researcher  
**I want** comprehensive evaluation metrics  
**So that** I can prove Event-based approach is superior

**Tasks**:
- [ ] Implement tournament system (Score vs Event, 500 games)
- [ ] Track metrics: win rate, convergence speed, game length
- [ ] Generate comparison plots (learning curves, win rate over time)
- [ ] Create statistical significance tests
- [ ] Write experimental results summary

**Acceptance Criteria**:
- Tournament results show Event-based > Score-based (>60% win rate)
- Learning curves show faster convergence for Event-based
- Results are statistically significant (p < 0.05)
- Figures ready for final report

---

## Technical Decisions Pending

1. **DQN vs PPO**: Which RL algorithm for Phase 1?
   - DQN: Simpler, off-policy, memory efficient
   - PPO: More stable, handles continuous action spaces better
   - **Recommendation**: Start with DQN for simplicity, can upgrade to PPO if needed

2. **State Representation**: How to encode game state?
   - Use existing state vectorization from modules/evaluators.py
   - Augment with additional features if needed

3. **Network Architecture**: Size and depth?
   - Start with 3-layer MLP (256-128-64)
   - Can upgrade to ResNet if needed (Phase 2)

4. **Self-Play Strategy**: How to generate training data?
   - Agent vs Random (initial training)
   - Agent vs Agent (self-play after 1k episodes)
   - Agent vs Strong Greedy (for harder challenges)

---

## Sprint Backlog (Task Board)

### To Do
- [ ] Create ADR for DQN vs PPO decision
- [ ] Implement state representation module
- [ ] Set up experiment logging (TensorBoard)
- [ ] Create score-based reward function
- [ ] Create event-based reward function

### In Progress
- [ ] Review existing evaluators and agent code

### Done
- [x] Project structure review
- [x] Development workflow documentation
- [x] Sprint 1 planning

---

## Definition of Done

A task is "Done" when:
1. Code is implemented and tested
2. Unit tests pass (if applicable)
3. Code is documented with docstrings
4. Development log is updated
5. Committed to git with meaningful message
6. Peer reviewed (for major features)

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Training takes longer than expected | High | Medium | Start early, use smaller networks if needed, parallelize self-play |
| Event-based doesn't outperform Score-based | High | Low | Carefully tune event weights, add more informative events |
| GPU memory issues | Medium | Low | Reduce batch size, use gradient accumulation |
| Team member unavailable | Medium | Low | Document everything, ensure knowledge sharing |

---

## Daily Logs

See `daily_logs/` subfolder for brief daily updates.

---

## Notes

- Remember to utilize all 32 CPU threads for self-play
- RTX 4090 should handle batch size 128-256 easily
- Target: 10k training episodes per agent
- Each game ~30-50 turns, ~5-10 sec per game = ~12-24 hours training time
