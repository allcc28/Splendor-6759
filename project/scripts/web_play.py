#!/usr/bin/env python3
"""
Web interface for playing Splendor against different agents.
Run with: python project/scripts/web_play.py
Then open http://localhost:5000 in browser.
"""

import os
import sys
import json
import subprocess
import importlib
import copy
import random
import threading
import shutil
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
from flask import Flask, render_template, request, jsonify

# Ensure child Python processes and Flask startup have valid locale settings.
os.environ.setdefault('LANG', 'en_US.UTF-8')
os.environ.setdefault('LC_ALL', 'en_US.UTF-8')
# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(project_dir)
modules_dir = os.path.join(project_root, "modules")
sys.path.insert(0, modules_dir)
sys.path.insert(0, project_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "project_event_based", "src"))
sys.path.insert(0, os.path.join(project_root, "project_event_based", "src", "utils"))

try:
    from state_vectorizer_event import vectorize_state_event as _event_vectorize
except ImportError:
    _event_vectorize = None

try:
    from event_detector import detect_events as _event_detect
except ImportError:
    _event_detect = None

from src.utils.splendor_gym_wrapper import SplendorGymWrapper
from gym_splendor_code.envs.splendor import SplendorEnv
from src.utils.state_vectorizer import SplendorStateVectorizer

app = Flask(__name__)

# Global variables for game state
game_env = None
ai_agent = None
vectorizer = None
legal_actions = None
game_done = False
winner = None
use_ai = True  # Toggle between AI and random opponent
final_round_active = False
opponent_type = 'value'
current_game_logged = False

# Cached opponent instances — created once per type, reused across games.
_opponent_cache = {}


class _PersistentInferenceClient:
    """Keep one inference subprocess alive and reuse loaded model across moves."""

    def __init__(self, python_cmd, inference_script, timeout_sec=2.5):
        self.python_cmd = python_cmd
        self.inference_script = inference_script
        self.timeout_sec = float(timeout_sec)
        self._proc = None
        self._lock = threading.Lock()
        self._stderr_pump_started = False

    def _start_stderr_pump(self):
        if self._stderr_pump_started or self._proc is None or self._proc.stderr is None:
            return

        def _pump():
            try:
                for line in self._proc.stderr:
                    txt = line.rstrip('\n')
                    if txt:
                        print(f'[infer-worker] {txt}')
            except Exception:
                pass

        threading.Thread(target=_pump, daemon=True).start()
        self._stderr_pump_started = True

    def _ensure_proc(self):
        if self._proc is not None and self._proc.poll() is None:
            return

        env = os.environ.copy()
        env.setdefault('LANG', 'en_US.UTF-8')
        env.setdefault('LC_ALL', 'en_US.UTF-8')
        env.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
        env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

        self._proc = subprocess.Popen(
            [self.python_cmd, self.inference_script, '--serve'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        self._stderr_pump_started = False
        self._start_stderr_pump()

    @staticmethod
    def _readline_with_timeout(pipe, timeout_sec):
        result = {}

        def _reader():
            try:
                result['line'] = pipe.readline()
            except Exception as exc:
                result['error'] = exc

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)

        if t.is_alive():
            raise TimeoutError(f'Inference worker response timeout ({timeout_sec}s)')
        if 'error' in result:
            raise result['error']

        line = result.get('line', '')
        if not line:
            raise RuntimeError('Inference worker returned empty response')
        return line

    def predict_action_idx(self, payload, timeout_sec=None):
        timeout = float(self.timeout_sec if timeout_sec is None else timeout_sec)
        with self._lock:
            self._ensure_proc()
            try:
                self._proc.stdin.write(json.dumps(payload) + '\n')
                self._proc.stdin.flush()
                deadline = datetime.now().timestamp() + timeout
                result = None

                while datetime.now().timestamp() < deadline:
                    remaining = max(0.05, deadline - datetime.now().timestamp())
                    line = self._readline_with_timeout(self._proc.stdout, remaining)
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        result = json.loads(stripped)
                        break
                    except json.JSONDecodeError:
                        # Ignore non-JSON noise emitted by native libraries.
                        continue

                if result is None:
                    raise TimeoutError(f'Inference worker did not return valid JSON within {timeout}s')

                if 'error' in result:
                    raise RuntimeError(result['error'])
                return int(result.get('action_idx', 0))
            except Exception:
                self.close()
                raise

    def close(self):
        if self._proc is None:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass
        self._proc = None


def _choose_fallback_action(legal_actions):
    """Choose a reasonable fallback action when model inference fails.

    Prefer card-buying actions so the opponent can still progress and score,
    then reserve, then gem trades, otherwise random legal action.
    """
    if not legal_actions:
        return None

    buy_actions = [a for a in legal_actions if getattr(a, 'action_type', None) == 'buy']
    if buy_actions:
        return random.choice(buy_actions)

    reserve_actions = [a for a in legal_actions if getattr(a, 'action_type', None) == 'reserve']
    if reserve_actions:
        return random.choice(reserve_actions)

    trade_actions = [a for a in legal_actions if getattr(a, 'action_type', None) == 'trade_gems']
    if trade_actions:
        return random.choice(trade_actions)

    return random.choice(legal_actions)


def _prefer_buy_when_token_capped(action, legal_actions, game_env_obj, token_cap=10):
    """When token count is capped, prefer a buy action to avoid stalling loops."""
    if game_env_obj is None or not legal_actions:
        return action

    buy_actions = [a for a in legal_actions if getattr(a, 'action_type', None) == 'buy']
    if not buy_actions:
        return action

    try:
        state = game_env_obj.current_state_of_the_game
        hand = state.list_of_players_hands[state.active_player_id]
        token_count = int(sum(hand.gems_possessed.to_dict()))
    except Exception:
        return action

    if token_count < int(token_cap):
        return action

    if action is None or getattr(action, 'action_type', None) != 'buy':
        return random.choice(buy_actions)
    return action


def _choose_ai_action_with_timeout(agent, game_env_obj, fallback_actions, timeout_sec=6.0):
    """Run AI action selection with a hard timeout so web requests never hang."""
    holder = {}

    def _worker():
        try:
            if hasattr(agent, 'choose_web_action'):
                holder['action'] = agent.choose_web_action(game_env_obj)
            elif hasattr(agent, 'choose_action'):
                observation = game_env_obj.show_observation('deterministic')
                holder['action'] = agent.choose_action(observation, [])
            else:
                holder['action'] = None
        except Exception as exc:
            holder['error'] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        print(f'Warning: AI action selection timed out after {timeout_sec}s; using fallback action.')
        fallback = _choose_fallback_action(fallback_actions)
        return _prefer_buy_when_token_capped(fallback, fallback_actions, game_env_obj)

    if holder.get('error') is not None:
        print(f"Warning: AI action selection failed ({holder['error']}); using fallback action.")
        fallback = _choose_fallback_action(fallback_actions)
        return _prefer_buy_when_token_capped(fallback, fallback_actions, game_env_obj)

    action = holder.get('action')
    if action is None:
        fallback = _choose_fallback_action(fallback_actions)
        return _prefer_buy_when_token_capped(fallback, fallback_actions, game_env_obj)
    return _prefer_buy_when_token_capped(action, fallback_actions, game_env_obj)


def _build_sb3_custom_objects(obs_size, action_space_size=200):
    """Build robust SB3 custom objects for cross-version model loading."""
    try:
        import gymnasium.spaces as gspaces
    except ImportError:
        import gym.spaces as gspaces

    return {
        'observation_space': gspaces.Box(low=0.0, high=1.0, shape=(int(obs_size),), dtype=np.float32),
        'action_space': gspaces.Discrete(int(action_space_size)),
        'clip_range': lambda _: 0.2,
        'lr_schedule': lambda _: 3e-4,
    }


def _install_numpy_compat_shims():
    """Install numpy compatibility shims required by some saved SB3 models."""
    # Models pickled under numpy 2.x can reference numpy._core.* on load.
    sys.modules.setdefault('numpy._core', np.core)

    submodule_names = [
        'numeric', 'multiarray', 'umath', 'fromnumeric', 'function_base',
        '_ufunc_config', '_dtype', '_type_aliases', 'arrayprint', 'shape_base',
        '_methods', 'overrides', 'records', 'defchararray', 'numerictypes',
        'einsumfunc', 'getlimits', 'memmap', 'machar', '_add_newdocs',
        '_add_newdocs_scalars', '_exceptions', '_internal', '_string_helpers',
    ]
    for name in submodule_names:
        dst_key = f'numpy._core.{name}'
        if dst_key not in sys.modules:
            src_key = f'numpy.core.{name}'
            if src_key in sys.modules:
                sys.modules[dst_key] = sys.modules[src_key]
            else:
                try:
                    sys.modules[dst_key] = importlib.import_module(src_key)
                except ImportError:
                    pass

    for key, val in list(sys.modules.items()):
        if key.startswith('numpy.core.') and ('numpy._core' + key[len('numpy.core'):]) not in sys.modules:
            sys.modules['numpy._core' + key[len('numpy.core'):]] = val

    # numpy random compat for models saved under newer numpy versions.
    try:
        import numpy.random._pickle as np_rand_pickle
        original_ctor = np_rand_pickle.__bit_generator_ctor

        def compat_bit_gen_ctor(bit_generator=None):
            if isinstance(bit_generator, type):
                return bit_generator()
            return original_ctor(bit_generator)

        np_rand_pickle.__bit_generator_ctor = compat_bit_gen_ctor
    except Exception:
        pass



def _find_inference_python():
    """Return the best Python interpreter for running inference subprocesses.

    Prefers /anaconda3/envs/splendor where models were trained and tested.
    Falls back to sys.executable if none of the known paths exist.
    """
    explicit = os.environ.get('SPLENDOR_INFERENCE_PYTHON')
    if explicit and os.path.isfile(explicit):
        return explicit

    candidates = []

    # Prefer the currently active environment first.
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, 'bin', 'python'))

    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        candidates.append(os.path.join(virtual_env, 'bin', 'python'))

    # Current interpreter is usually the safest cross-machine default.
    candidates.append(sys.executable)

    # Then try interpreters visible from PATH.
    for cmd in ('python3', 'python'):
        p = shutil.which(cmd)
        if p:
            candidates.append(p)

    # Keep previous machine-specific paths only as a last-resort fallback.
    candidates = [
        *candidates,
        os.path.expanduser('~/anaconda3/envs/splendor_event311/bin/python'),
        os.path.expanduser('~/anaconda3/envs/splendor/bin/python'),
        os.path.expanduser('~/miniconda3/envs/splendor/bin/python'),
    ]

    seen = set()
    for p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return p

    return sys.executable


def _supports_float_schedule(python_cmd: str) -> bool:
    """Check whether the interpreter has a SB3 build that provides FloatSchedule."""
    if not python_cmd or not os.path.isfile(python_cmd):
        return False
    try:
        completed = subprocess.run(
            [python_cmd, '-c', 'from stable_baselines3.common.utils import FloatSchedule'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _find_event_inference_python():
    """Resolve interpreter for event-model inference with SB3 compatibility checks."""
    preferred = os.environ.get('EVENT_AGENT_PYTHON')
    if preferred and os.path.isfile(preferred):
        return preferred

    shared = os.environ.get('SPLENDOR_INFERENCE_PYTHON')
    if shared and os.path.isfile(shared) and _supports_float_schedule(shared):
        return shared

    candidates = []
    # Try common event env names first.
    candidates.extend([
        os.path.expanduser('~/anaconda3/envs/splendor_event311/bin/python'),
        os.path.expanduser('~/miniconda3/envs/splendor_event311/bin/python'),
    ])

    # Fall back to generic resolution order.
    candidates.append(_find_inference_python())

    seen = set()
    for p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)
        if _supports_float_schedule(p):
            return p

    return _find_inference_python()


def _try_load_cached_model(model_path, model_kind, obs_size=135, action_space_size=200):
    """Try loading model once in current process for low-latency inference."""
    _install_numpy_compat_shims()
    custom_objects = _build_sb3_custom_objects(obs_size=obs_size, action_space_size=action_space_size)
    if model_kind == 'maskable':
        from sb3_contrib import MaskablePPO
        try:
            # Keep native policy/settings when compatible; this best preserves logic.
            return MaskablePPO.load(model_path)
        except Exception:
            return MaskablePPO.load(model_path, custom_objects=custom_objects)

    from stable_baselines3 import PPO
    try:
        return PPO.load(model_path)
    except Exception:
        return PPO.load(model_path, custom_objects=custom_objects)


def _resolve_value_model_info():
    """Resolve the score-based PPO model to use for the web value agent.

    Preference order:
    1. Model artifact matching project/configs/training/maskable_ppo_v4a_ent_lr.yaml
    2. Legacy PPO v1 final model as fallback if maskable artifact is absent
    """
    config_path = os.path.join(project_root, 'project', 'configs', 'training', 'maskable_ppo_v4a_ent_lr.yaml')
    root = Path(project_root)

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        experiment_name = cfg.get('experiment', {}).get('name')
        if experiment_name:
            candidates = []
            candidates.extend(root.glob(f'project/logs/{experiment_name}_*/eval/best_model.zip'))
            candidates.extend(root.glob(f'project/logs/{experiment_name}_*/final_model.zip'))
            if candidates:
                candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
                return {
                    'model_path': str(candidates[0]),
                    'model_kind': 'maskable',
                    'model_source': 'maskable_ppo_v4a_ent_lr.yaml',
                }

    fallback = root / 'project' / 'logs' / 'ppo_score_based_v1_20260224_113524' / 'final_model.zip'
    if fallback.exists():
        return {
            'model_path': str(fallback),
            'model_kind': 'ppo',
            'model_source': 'fallback_ppo_score_based_v1',
        }

    raise FileNotFoundError(
        'No score-based PPO model found. Expected a maskable model from '
        'project/configs/training/maskable_ppo_v4a_ent_lr.yaml or the fallback '
        'project/logs/ppo_score_based_v1_20260224_113524/final_model.zip'
    )


class ScoreBasedPPOOpponent:
    """Web opponent adapter for trained score-based PPO / MaskablePPO models."""

    def __init__(self):
        model_info = _resolve_value_model_info()
        self.model_path = model_info['model_path']
        self.model_kind = model_info['model_kind']
        self.model_source = model_info['model_source']
        self.vectorizer = SplendorStateVectorizer()
        self.inference_script = os.path.join(project_root, 'project', 'scripts', 'web_score_inference.py')
        self.python_cmd = os.environ.get('VALUE_AGENT_PYTHON', _find_inference_python())
        self.infer_client = _PersistentInferenceClient(
            python_cmd=self.python_cmd,
            inference_script=self.inference_script,
            timeout_sec=2.8,
        )
        self.cached_model = None
        if os.environ.get('WEB_INPROCESS_FASTPATH', '').lower() in ('1', 'true', 'yes'):
            self._init_cached_model()
        self._warm_up_worker()

    def _warm_up_worker(self):
        """Warm up persistent inference process in the background (non-blocking).

        Uses a temporary client so warm-up model loading does not contend on the
        live inference client lock used by gameplay requests.
        """
        def _do_warmup():
            warm_client = _PersistentInferenceClient(
                python_cmd=self.python_cmd,
                inference_script=self.inference_script,
                timeout_sec=self.infer_client.timeout_sec,
            )
            payload = {
                'model_path': self.model_path,
                'model_kind': self.model_kind,
                'observation': np.zeros(135, dtype=np.float32).tolist(),
                'legal_actions_count': 1,
            }
            try:
                warm_client.predict_action_idx(payload, timeout_sec=40.0)
                old_client = self.infer_client
                self.infer_client = warm_client
                old_client.close()
                print('ScoreBasedPPOOpponent: persistent inference worker warmed up.')
            except Exception as exc:
                warm_client.close()
                print(f'Warning: value worker warmup failed ({exc}); will use fallback when needed.')
        threading.Thread(target=_do_warmup, daemon=True).start()

    def _init_cached_model(self):
        """Warm-load model once to avoid per-step subprocess+load overhead."""
        try:
            os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
            os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
            self.cached_model = _try_load_cached_model(
                model_path=self.model_path,
                model_kind=self.model_kind,
                obs_size=135,
                action_space_size=200,
            )
            print(f'ScoreBasedPPOOpponent: in-process model loaded from {self.model_path}')
        except Exception as exc:
            self.cached_model = None
            print(f'Warning: in-process value model load failed ({exc}); fallback to subprocess inference.')

    def _predict_action_idx(self, obs, legal_actions_count):
        """Predict action index using fast in-process model when available."""
        if self.cached_model is None:
            return None

        if self.model_kind == 'maskable':
            action_mask = np.zeros(200, dtype=bool)
            action_mask[:max(0, min(int(legal_actions_count), 200))] = True
            action, _ = self.cached_model.predict(obs, action_masks=action_mask, deterministic=True)
        else:
            action, _ = self.cached_model.predict(obs, deterministic=True)

        return int(np.asarray(action).squeeze())

    def choose_web_action(self, game_env_obj):
        game_env_obj.update_actions()
        legal_actions = game_env_obj.action_space.list_of_actions
        if not legal_actions:
            return None

        state = game_env_obj.current_state_of_the_game
        obs = self.vectorizer.vectorize(state, player_id=state.active_player_id, turn_count=0)

        # Fast path: in-process model prediction (same model, no logic change).
        try:
            action_idx = self._predict_action_idx(obs, len(legal_actions))
            if action_idx is not None and 0 <= action_idx < len(legal_actions):
                return legal_actions[action_idx]
        except Exception as exc:
            print(f'Warning: in-process value prediction failed ({exc}); fallback to subprocess.')

        payload = {
            'model_path': self.model_path,
            'model_kind': self.model_kind,
            'observation': obs.tolist(),
            'legal_actions_count': len(legal_actions),
        }

        # Fast path: persistent inference worker with model cache in subprocess.
        try:
            action_idx = self.infer_client.predict_action_idx(payload)
            if 0 <= action_idx < len(legal_actions):
                return legal_actions[action_idx]
        except Exception as exc:
            print(f'Warning: persistent value inference failed ({exc}); fallback to one-shot subprocess.')

        env = os.environ.copy()
        env.setdefault('LANG', 'en_US.UTF-8')
        env.setdefault('LC_ALL', 'en_US.UTF-8')
        env.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

        try:
            completed = subprocess.run(
                [self.python_cmd, self.inference_script],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=True,
                env=env,
                timeout=25,
            )
            result = json.loads(completed.stdout)
            action_idx = int(result.get('action_idx', 0))
        except Exception as exc:
            print(f'Warning: score PPO inference failed ({exc}); using heuristic fallback action.')
            return _choose_fallback_action(legal_actions)

        if action_idx < 0 or action_idx >= len(legal_actions):
            return _choose_fallback_action(legal_actions)

        return legal_actions[action_idx]





def _resolve_event_model_info():
    """Use the fixed v3_1m_1000000_steps.zip model for the event-based opponent.

    This artifact is loaded via PPO (non-maskable) for compatibility.
    """
    root = Path(project_root)
    model_path = root / 'project_event_based' / 'notebooks' / 'models' / 'v3_1m_1000000_steps.zip'
    if not model_path.exists():
        raise FileNotFoundError(f'Event model not found: {model_path}')
    return {
        'model_path': str(model_path),
        'model_kind': 'maskable',
        'model_source': 'v3_1m_1000000_steps',
    }


class EventBasedPPOOpponent:
    """Web opponent using the trained event-based PPO model via subprocess inference."""

    def __init__(self):
        model_info = _resolve_event_model_info()
        self.model_path = model_info['model_path']
        self.model_kind = model_info['model_kind']
        self.inference_script = os.path.join(project_root, 'project', 'scripts', 'web_score_inference.py')
        self.python_cmd = _find_event_inference_python()
        self.vectorizer = SplendorStateVectorizer()
        # Keep the same event-history signal used during training.
        self.last_event = np.zeros(9, dtype=np.float32)
        self.infer_client = _PersistentInferenceClient(
            python_cmd=self.python_cmd,
            inference_script=self.inference_script,
            timeout_sec=4,
        )
        self.cached_model = None
        if os.environ.get('WEB_INPROCESS_FASTPATH', '').lower() in ('1', 'true', 'yes'):
            self._init_cached_model()
        # Disable proactive warm-up for event model: loading can be heavy and
        # hurts first-turn responsiveness when it contends with live requests.
        print(f'EventBasedPPOOpponent: model={self.model_path}')

    def reset_context(self):
        """Reset per-game recurrent context (event history tail)."""
        self.last_event = np.zeros(9, dtype=np.float32)

    def _warm_up_worker(self):
        """Warm up persistent inference process in the background (non-blocking).

        Uses a temporary client so warm-up model loading does not contend on the
        live inference client lock used by gameplay requests.
        """
        def _do_warmup():
            warm_client = _PersistentInferenceClient(
                python_cmd=self.python_cmd,
                inference_script=self.inference_script,
                timeout_sec=self.infer_client.timeout_sec,
            )
            payload = {
                'model_path': self.model_path,
                'model_kind': self.model_kind,
                'observation': np.zeros(109, dtype=np.float32).tolist(),
                'legal_actions_count': 1,
            }
            try:
                warm_client.predict_action_idx(payload, timeout_sec=40.0)
                old_client = self.infer_client
                self.infer_client = warm_client
                old_client.close()
                print('EventBasedPPOOpponent: persistent inference worker warmed up.')
            except Exception as exc:
                warm_client.close()
                print(f'Warning: event worker warmup failed ({exc}); will use fallback when needed.')
        threading.Thread(target=_do_warmup, daemon=True).start()

    def _init_cached_model(self):
        """Warm-load event model once for low-latency online play."""
        try:
            os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
            os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
            self.cached_model = _try_load_cached_model(
                model_path=self.model_path,
                model_kind=self.model_kind,
                obs_size=109,
                action_space_size=200,
            )
            print(f'EventBasedPPOOpponent: in-process model loaded from {self.model_path}')
        except Exception as exc:
            self.cached_model = None
            print(f'Warning: in-process event model load failed ({exc}); fallback to subprocess inference.')

    def _predict_action_idx(self, obs, legal_actions_count):
        """Predict action index using fast in-process model when available."""
        if self.cached_model is None:
            return None

        if self.model_kind == 'maskable':
            action_mask = np.zeros(200, dtype=bool)
            action_mask[:max(0, min(int(legal_actions_count), 200))] = True
            action, _ = self.cached_model.predict(obs, action_masks=action_mask, deterministic=True)
        else:
            action, _ = self.cached_model.predict(obs, deterministic=True)

        return int(np.asarray(action).squeeze())

    def _build_raw_obs135(self, state):
        """Build the base 135-d observation from the active-player perspective."""
        actor_id = state.active_player_id
        try:
            raw = self.vectorizer.vectorize(state, player_id=actor_id, turn_count=0)
            return np.asarray(raw, dtype=np.float32).reshape(-1)[:135]
        except Exception:
            return np.zeros(135, dtype=np.float32)

    def _get_gem_gap_features(self, raw_obs135):
        """Match train_maskable.EventRewardWrapper gap-feature construction (60 dims)."""
        obs = np.asarray(raw_obs135, dtype=np.float32).reshape(-1)
        player_gems = obs[0:5] if obs.shape[0] >= 5 else np.zeros(5, dtype=np.float32)
        player_discounts = obs[5:10] if obs.shape[0] >= 10 else np.zeros(5, dtype=np.float32)
        total_assets = player_gems + player_discounts

        gaps = []
        cards_offset = 40
        for i in range(12):
            card_start = cards_offset + (i * 5)
            if card_start + 5 <= obs.shape[0]:
                card_cost = obs[card_start: card_start + 5]
                gap = np.maximum(0.0, card_cost - total_assets)
                gaps.extend(gap.tolist())
            else:
                gaps.extend([0.0] * 5)

        if len(gaps) < 60:
            gaps.extend([0.0] * (60 - len(gaps)))
        return np.asarray(gaps[:60], dtype=np.float32)

    def _build_observation(self, state):
        """Build event observation matching this model's expected 109 dims.

        Format: 40 state dims + 60 gap dims + 9 event-history dims.
        """
        raw135 = self._build_raw_obs135(state)
        state40 = np.asarray(raw135[:40], dtype=np.float32)
        gap60 = self._get_gem_gap_features(raw135)
        return np.concatenate([state40, gap60, self.last_event]).astype(np.float32)

    def _update_last_event(self, state_before_action, chosen_action):
        """Update the 9-d event tail to mirror EventRewardWrapper behavior."""
        if _event_detect is None:
            self.last_event = np.zeros(9, dtype=np.float32)
            return

        prev_vec = self._build_raw_obs135(state_before_action)[:40]

        try:
            state_copy = copy.deepcopy(state_before_action)
            chosen_action.execute(state_copy)
            next_vec = self._build_raw_obs135(state_copy)[:40]
            ev = _event_detect(prev_vec, chosen_action, next_vec)
            self.last_event = np.asarray(ev, dtype=np.float32)[:9]
        except Exception:
            self.last_event = np.zeros(9, dtype=np.float32)

    def choose_web_action(self, game_env_obj):
        game_env_obj.update_actions()
        legal_actions = game_env_obj.action_space.list_of_actions
        if not legal_actions:
            return None

        state = game_env_obj.current_state_of_the_game
        obs = self._build_observation(state)

        # Fast path: in-process model prediction (same model, no logic change).
        try:
            action_idx = self._predict_action_idx(obs, len(legal_actions))
            if action_idx is not None and 0 <= action_idx < len(legal_actions):
                chosen = legal_actions[action_idx]
                self._update_last_event(state, chosen)
                return chosen
        except Exception as exc:
            print(f'Warning: in-process event prediction failed ({exc}); fallback to subprocess.')

        payload = {
            'model_path': self.model_path,
            'model_kind': self.model_kind,
            'observation': obs.tolist(),
            'legal_actions_count': len(legal_actions),
        }

        # Fast path: persistent inference worker with model cache in subprocess.
        try:
            action_idx = self.infer_client.predict_action_idx(payload)
            if 0 <= action_idx < len(legal_actions):
                chosen = legal_actions[action_idx]
                self._update_last_event(state, chosen)
                return chosen
        except Exception as exc:
            print(f'Warning: persistent event inference failed ({exc}); using heuristic fallback action.')
            self.last_event = np.zeros(9, dtype=np.float32)
            return _choose_fallback_action(legal_actions)


def _get_opponent_label():
    """Human-readable opponent label for UI and logs."""
    mapping = {
        'value': 'Value-based Agent',
        'event': 'Event-based Agent',
        'random': 'Random Agent',
    }
    return mapping.get(opponent_type, 'Value-based Agent')


def _get_match_result():
    """Compute final winner by score, then card count as tie-breaker."""
    state = game_env.current_state_of_the_game
    human_hand = state.list_of_players_hands[0]
    opp_hand = state.list_of_players_hands[1]

    human_score = human_hand.number_of_my_points()
    opp_score = opp_hand.number_of_my_points()
    human_cards = len(human_hand.cards_possessed)
    opp_cards = len(opp_hand.cards_possessed)

    if human_score > opp_score:
        winner_id = 'human'
        reason = 'score'
    elif human_score < opp_score:
        winner_id = 'ai'
        reason = 'score'
    elif human_cards > opp_cards:
        winner_id = 'human'
        reason = 'cards'
    elif human_cards < opp_cards:
        winner_id = 'ai'
        reason = 'cards'
    else:
        winner_id = 'draw'
        reason = 'draw'

    return {
        'resolved_winner': winner_id,
        'winner_reason': reason,
        'final_scores': {'human': human_score, 'ai': opp_score},
        'final_card_counts': {'human': human_cards, 'ai': opp_cards},
    }


def _update_final_round_status():
    """Activate final round at 15+ points; end when round returns to player 0."""
    global game_done, winner, final_round_active

    if game_done:
        return

    state = game_env.current_state_of_the_game
    human_score = state.list_of_players_hands[0].number_of_my_points()
    opp_score = state.list_of_players_hands[1].number_of_my_points()

    if not final_round_active and (human_score >= 15 or opp_score >= 15):
        final_round_active = True

    # In 2-player game with player 0 starting, active_player_id == 0 means round boundary reached.
    if final_round_active and state.active_player_id == 0:
        result = _get_match_result()
        game_done = True
        winner = result['resolved_winner']


def _persist_game_result_if_needed():
    """Append one completed game result to persistent history file."""
    global current_game_logged

    if not game_done or current_game_logged or game_env is None:
        return

    result = _get_match_result()
    history_record = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'opponent_type': opponent_type,
        'opponent_label': _get_opponent_label(),
        'winner': result['resolved_winner'],
        'winner_reason': result['winner_reason'],
        'final_scores': result['final_scores'],
        'final_card_counts': result['final_card_counts'],
    }

    history_file = os.path.join(project_dir, 'outputs', 'web_game_history.jsonl')
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(history_record, ensure_ascii=True) + '\n')

    current_game_logged = True

def init_game(selected_opponent_type='value'):
    """Initialize a new game"""
    global game_env, ai_agent, vectorizer, legal_actions, game_done, winner, use_ai, final_round_active, opponent_type, current_game_logged

    opponent_type = selected_opponent_type if selected_opponent_type in ('value', 'event', 'random') else 'value'
    use_ai = opponent_type in ('value', 'event')

    # Initialize vectorizer (used for debugging / future model integration)
    vectorizer = SplendorStateVectorizer()

    # Create environment
    game_env = SplendorEnv()

    # Choose opponent — reuse cached instance to avoid re-loading model on every game.
    global _opponent_cache
    if opponent_type in ('event', 'value'):
        if opponent_type not in _opponent_cache:
            try:
                if opponent_type == 'event':
                    _opponent_cache['event'] = EventBasedPPOOpponent()
                else:
                    _opponent_cache['value'] = ScoreBasedPPOOpponent()
            except Exception as exc:
                print(f'Warning: could not load {opponent_type} PPO model ({exc}); falling back to random opponent.')
                _opponent_cache[opponent_type] = None
        ai_agent = _opponent_cache[opponent_type]
        if ai_agent is not None and hasattr(ai_agent, 'reset_context'):
            try:
                ai_agent.reset_context()
            except Exception:
                pass
    else:
        ai_agent = None

    # Reset game
    game_env.reset()
    legal_actions = game_env.action_space.list_of_actions
    game_done = False
    winner = None
    final_round_active = False
    current_game_logged = False

    return get_game_state()

def get_game_state():
    """Get current game state for frontend"""
    if game_env is None:
        return None

    state = game_env.current_state_of_the_game

    # Get player hands
    player_hand = state.list_of_players_hands[0]  # Human player
    ai_hand = state.list_of_players_hands[1]      # AI player

    # Get board state
    board = state.board

    # Convert gems dict to proper format
    def gems_to_dict(gems_collection):
        gems_list = gems_collection.to_dict()
        return {
            'gold': gems_list[0],
            'red': gems_list[1],
            'green': gems_list[2],
            'blue': gems_list[3],
            'white': gems_list[4],
            'black': gems_list[5]
        }

    # Convert cards to list
    def cards_to_list(cards):
        return [{
            'id': card.id,
            'level': int(card.row.value) + 1,
            'color': card.discount_profit.name.lower(),
            'points': card.victory_points,
            'cost': gems_to_dict(card.price)
        } for card in cards]

    # Board cards are stored as a set; group by row so we can display them by level
    cards_by_row = {0: [], 1: [], 2: []}
    for card in board.cards_on_board:
        row_idx = int(card.row.value) if hasattr(card.row, 'value') else int(card.row)
        if row_idx in cards_by_row:
            cards_by_row[row_idx].append(card)

    return {
        'player_gems': gems_to_dict(player_hand.gems_possessed),
        'player_cards': cards_to_list(player_hand.cards_possessed),
        'player_reserved': cards_to_list(player_hand.cards_reserved),
        'player_score': player_hand.number_of_my_points(),
        'ai_gems': gems_to_dict(ai_hand.gems_possessed),
        'ai_cards': cards_to_list(ai_hand.cards_possessed),  # Now return full card list for display
        'ai_score': ai_hand.number_of_my_points(),
        'board_gems': gems_to_dict(board.gems_on_board),
        'board_cards': {
            'level1': cards_to_list(cards_by_row[0]),
            'level2': cards_to_list(cards_by_row[1]),
            'level3': cards_to_list(cards_by_row[2])
        },
        'nobles': [{
            'requirements': gems_to_dict(noble.price),
            'points': noble.victory_points
        } for noble in board.nobles_on_board],
        'legal_actions': [action.to_dict() for action in legal_actions],
        'game_done': game_done,
        'winner': winner,
        'opponent_type': opponent_type,
        'opponent_label': _get_opponent_label(),
        **_get_match_result(),
        'current_player': 'human' if state.active_player_id == 0 else 'ai'
    }

@app.route('/')
def index():
    """Serve the main game page"""
    return render_template('game.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    """Start a new game"""
    data = request.get_json() or {}
    selected_opponent_type = data.get('opponent_type')

    # Backward compatibility for old frontend payload
    if selected_opponent_type is None:
        use_ai_flag = data.get('use_ai', True)
        selected_opponent_type = 'value' if use_ai_flag else 'random'

    state = init_game(selected_opponent_type)
    return jsonify(state)

@app.route('/make_move', methods=['POST'])
def make_move():
    """Handle human player's move"""
    global legal_actions, game_done, winner

    if game_done:
        return jsonify({'error': 'Game is already finished'})

    data = request.get_json()
    action_idx = data.get('action_idx')

    if action_idx is None or action_idx < 0 or action_idx >= len(legal_actions):
        return jsonify({'error': 'Invalid action index'})

    # Execute human move
    action = legal_actions[action_idx]
    obs, reward, done, info = game_env.step('deterministic', action, return_observation=False)

    # Update legal actions based on the new state
    game_env.update_actions()
    legal_actions = game_env.action_space.list_of_actions
    _update_final_round_status()

    if game_done:
        _persist_game_result_if_needed()
        return jsonify(get_game_state())

    # AI's turn
    if use_ai and ai_agent is not None:
        ai_action = _choose_ai_action_with_timeout(ai_agent, game_env, legal_actions, timeout_sec=8.0)
    else:
        # Random opponent
        import random
        ai_action = random.choice(legal_actions) if legal_actions else None

    if ai_action is None:
        # No action possible
        game_done = True
        winner = 'human'
    else:
        obs, reward, done, info = game_env.step('deterministic', ai_action, return_observation=False)
        # Update legal actions after AI move
        game_env.update_actions()
        _update_final_round_status()

    _persist_game_result_if_needed()

    # Update legal actions for next human turn
    legal_actions = game_env.action_space.list_of_actions

    return jsonify(get_game_state())

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)

    # Create static directory for CSS/JS
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False)