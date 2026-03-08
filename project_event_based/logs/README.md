Project event-based training logs — artifact guide

This folder contains per-experiment subfolders created by the event-based
training script. Each experiment folder follows the pattern:

  <exp_name>_YYYYMMDD_HHMMSS/

Common artifacts inside an experiment folder:

- `final_model` or `final_model.zip`  : Stable-Baselines3 saved model (use `PPO.load(...)`).
- `interrupted_model`                 : Model saved when training was interrupted.
- `checkpoints/`                      : Periodic checkpoints saved by `CheckpointCallback`.
- `eval/best_model`                   : Best model saved during evaluation (if EvalCallback used).
- `vecnormalize.pkl`                  : If `VecNormalize` was used, this file stores obs/reward normalization stats.
- `tensorboard/`                      : TensorBoard logs (pass this to `tensorboard --logdir`).
- `monitor/`                          : Gym Monitor logs and episode-level traces.
- `summary.json`                      : Brief experiment summary (paths, timestamp, total_timesteps).

Quick inspect & load examples:

List latest experiments:

  ls -1 project_event_based/logs | tail -n 10

Load a saved model in Python:

  from stable_baselines3 import PPO
  model = PPO.load('project_event_based/logs/<exp>/final_model')

Load `VecNormalize` statistics (if present):

  from stable_baselines3.common.vec_env import VecNormalize
  vecnorm = VecNormalize.load('project_event_based/logs/<exp>/vecnormalize.pkl', None)

Run TensorBoard for all event-based runs:

  tensorboard --logdir project_event_based/logs --host 0.0.0.0 --port 6006

Notes
- The training script also attempts to create a `project_event_based/logs/<exp_name>_latest`
  symlink pointing to the most recent run for convenience.
- If you resume from a checkpoint, pass the checkpoint path to the training script `--resume`.
