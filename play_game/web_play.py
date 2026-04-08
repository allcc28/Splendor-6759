#!/usr/bin/env python3
"""
Web interface for playing Splendor against different agents.
Run with: python project/scripts/web_play.py
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
project_root = os.path.dirname(script_dir)  # play_game/ sits directly under repo root
modules_dir = os.path.join(project_root, "modules")
sys.path.insert(0, modules_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "project"))
sys.path.insert(0, os.path.join(project_root, "project_event_based"))
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


def _state_to_event_training_raw135(state_obj, player_id: int):
    """135-d raw obs matching project_event_based SplendorGymWrapper fallback (40-d event vec + zero pad).

    This mirrors ``splendor_gym_wrapper.SplendorStateVectorizer`` when ``state_vectorizer`` is absent:
    first 40 dims from ``vectorize_state_event``, indices 40+ are zeros (same as training pipeline).
    """
    if _event_vectorize is None:
        return np.zeros(135, dtype=np.float32)

    board_gems = [0] * 6
    try:
        bg = getattr(state_obj.board, 'gems_on_board', None)
        if bg is None:
            bg = getattr(state_obj.board, 'tokens_on_board', None)
        if bg is not None:
            board_gems = list(bg)[:6]
    except Exception:
        pass

    players = []
    try:
        hands = list(state_obj.list_of_players_hands)
    except Exception:
        hands = []
    for p in hands:
        try:
            score = p.number_of_my_points()
        except Exception:
            score = getattr(p, 'score', 0)
        try:
            gems = list(getattr(p, 'gems_on_hand', getattr(p, 'tokens', [0] * 6)))[:6]
        except Exception:
            gems = [0] * 6
        try:
            discounts = list(getattr(p, 'discounts', getattr(p, 'permanent_discounts', [0] * 6)))[:6]
        except Exception:
            discounts = [0] * 6
        try:
            reserved = int(len(getattr(p, 'reserved_cards', [])))
        except Exception:
            reserved = int(getattr(p, 'reserved_count', 0))
        players.append({'score': score, 'gems': gems, 'discounts': discounts, 'reserved_count': reserved})
    while len(players) < 2:
        players.append({'score': 0, 'gems': [0] * 6, 'discounts': [0] * 6, 'reserved_count': 0})

    state_dict = {'board': {'gems': board_gems}, 'players': players}
    vec = np.zeros(135, dtype=np.float32)
    vec[:40] = np.asarray(_event_vectorize(state_dict, active_player_index=int(player_id)), dtype=np.float32).reshape(-1)[:40]
    return vec.astype(np.float32)

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
        timeout = self.timeout_sec if timeout_sec is None else timeout_sec
        with self._lock:
            self._ensure_proc()
            try:
                self._proc.stdin.write(json.dumps(payload) + '\n')
                self._proc.stdin.flush()
                result = None

                # timeout <= 0 means wait indefinitely (used for the first event move).
                if timeout is None or float(timeout) <= 0:
                    while True:
                        # Truly unbounded wait: block on one line instead of applying
                        # a hidden fixed timeout chunk.
                        line = self._proc.stdout.readline()
                        if not line:
                            if self._proc.poll() is not None:
                                raise RuntimeError('Inference worker exited while waiting for response')
                            continue
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            result = json.loads(stripped)
                            break
                        except json.JSONDecodeError:
                            # Ignore non-JSON noise emitted by native libraries.
                            continue
                else:
                    timeout = float(timeout)
                    deadline = datetime.now().timestamp() + timeout
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
    if timeout_sec is None or float(timeout_sec) <= 0:
        t.join()
    else:
        t.join(timeout=float(timeout_sec))

    if t.is_alive():
        raise TimeoutError(f'AI action selection timed out after {timeout_sec}s')

    if holder.get('error') is not None:
        raise RuntimeError(f"AI action selection failed: {holder['error']}") from holder['error']

    action = holder.get('action')
    if action is None:
        raise RuntimeError('AI action selection returned no action')
    return action


def _parse_timeout_env(name: str, default_value: float) -> float:
    """Parse timeout env var; fallback to default on invalid values."""
    raw = os.environ.get(name)
    if raw is None:
        return float(default_value)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default_value)


def _resolve_ai_action_timeout(agent) -> float:
    """Resolve per-turn AI timeout with optional unlimited first event move."""
    regular_timeout = _parse_timeout_env('AI_ACTION_TIMEOUT_SEC', 45.0)
    first_event_timeout = _parse_timeout_env('AI_FIRST_EVENT_TIMEOUT_SEC', 0.0)

    # Event model may need a long cold start; first move should not be truncated.
    if hasattr(agent, '_first_move_pending') and bool(getattr(agent, '_first_move_pending')):
        return first_event_timeout

    return regular_timeout


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


def _install_sb3_compat_shims():
    """Install SB3 compatibility shims required by newer saved models."""
    try:
        from stable_baselines3.common import utils as sb3_utils

        if not hasattr(sb3_utils, 'FloatSchedule'):
            class FloatSchedule:
                def __init__(self, value_schedule):
                    self.value_schedule = value_schedule

                def __call__(self, progress_remaining):
                    return float(self.value_schedule(progress_remaining))

            sb3_utils.FloatSchedule = FloatSchedule
    except Exception:
        pass


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
    Falls back to sys.executable if none of the known paths exist.
    """
    explicit = os.environ.get('SPLENDOR_INFERENCE_PYTHON')
    if explicit and os.path.isfile(explicit):
        return explicit

    candidates = []

    # Current interpreter is usually the safest cross-machine default.
    candidates.append(sys.executable)

    # Prefer the currently active environment when available.
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, 'bin', 'python'))

    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        candidates.append(os.path.join(virtual_env, 'bin', 'python'))

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


def _discover_conda_python_candidates():
    """Best-effort discovery of python executables from local conda envs."""
    conda_cmd = os.environ.get('CONDA_EXE') or shutil.which('conda')
    if not conda_cmd:
        return []

    try:
        completed = subprocess.run(
            [conda_cmd, 'env', 'list', '--json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout:
            return []

        payload = json.loads(completed.stdout)
        env_paths = payload.get('envs') or []
        candidates = []
        for env_path in env_paths:
            if not env_path:
                continue
            py = os.path.join(env_path, 'bin', 'python')
            if os.path.isfile(py):
                candidates.append(py)
        return candidates
    except Exception:
        return []


def _supports_float_schedule(python_cmd: str) -> bool:
    """Check whether an interpreter can support our SB3 FloatSchedule load path."""
    if not python_cmd or not os.path.isfile(python_cmd):
        return False

    probe_code = (
        'import sys\n'
        'try:\n'
        '    from stable_baselines3.common import utils as sb3_utils\n'
        '    import sb3_contrib\n'
        'except Exception:\n'
        '    raise SystemExit(1)\n'
        'if not hasattr(sb3_utils, "FloatSchedule"):\n'
        '    class FloatSchedule:\n'
        '        def __init__(self, value_schedule):\n'
        '            self.value_schedule = value_schedule\n'
        '        def __call__(self, progress_remaining):\n'
        '            return float(self.value_schedule(progress_remaining))\n'
        '    sb3_utils.FloatSchedule = FloatSchedule\n'
        'raise SystemExit(0)\n'
    )

    try:
        completed = subprocess.run(
            [python_cmd, '-c', probe_code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _supports_event_inference_runtime(python_cmd: str, model_path: str) -> bool:
    """Check whether an interpreter can run a minimal event inference end-to-end."""
    if not python_cmd or not os.path.isfile(python_cmd):
        return False
    if not model_path or not os.path.isfile(model_path):
        return False

    script = os.path.join(script_dir, 'web_score_inference.py')
    if not os.path.isfile(script):
        return False

    payload = {
        'model_path': model_path,
        'model_kind': 'maskable',
        'observation': np.zeros(109, dtype=np.float32).tolist(),
        'legal_actions_count': 1,
    }
    env = os.environ.copy()
    env.setdefault('LANG', 'en_US.UTF-8')
    env.setdefault('LC_ALL', 'en_US.UTF-8')
    env.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    try:
        completed = subprocess.run(
            [python_cmd, script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=25,
            check=False,
            env=env,
        )
        if completed.returncode != 0 or not completed.stdout:
            return False
        data = json.loads(completed.stdout)
        action_idx = int(data.get('action_idx', -1))
        return action_idx >= 0
    except Exception:
        return False


def _find_event_inference_python(model_path: str = None):
    """Resolve interpreter for event-model inference without env-name assumptions."""
    preferred = os.environ.get('EVENT_AGENT_PYTHON')
    if preferred and os.path.isfile(preferred) and _supports_float_schedule(preferred):
        if model_path is None or _supports_event_inference_runtime(preferred, model_path):
            return preferred

    candidates = []

    shared = os.environ.get('SPLENDOR_INFERENCE_PYTHON')
    if shared:
        candidates.append(shared)

    # Optional explicit list for teammates (path separator: ':' on macOS/Linux, ';' on Windows).
    extra = os.environ.get('EVENT_AGENT_PYTHONS', '')
    if extra:
        candidates.extend([p.strip() for p in extra.split(os.pathsep) if p.strip()])

    # Prefer currently active interpreter/env first.
    candidates.append(sys.executable)

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, 'bin', 'python'))

    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        candidates.append(os.path.join(virtual_env, 'bin', 'python'))

    for cmd in ('python3', 'python'):
        p = shutil.which(cmd)
        if p:
            candidates.append(p)

    # Discover all conda envs dynamically so env names can differ across machines.
    candidates.extend(_discover_conda_python_candidates())

    # Keep generic fallback for continuity.
    candidates.append(_find_inference_python())

    # Runtime health-check can be expensive; cap candidate attempts.
    max_candidates = 8
    seen = set()
    checked = 0
    for p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)
        if not _supports_float_schedule(p):
            continue

        if model_path is None:
            return p

        checked += 1
        if _supports_event_inference_runtime(p, model_path):
            return p
        if checked >= max_candidates:
            break

    return _find_inference_python()


def _find_value_inference_python():
    """Resolve interpreter for score-model inference with SB3 compatibility checks."""
    preferred = os.environ.get('VALUE_AGENT_PYTHON')
    if preferred and os.path.isfile(preferred):
        return preferred

    known_good = os.path.expanduser('~/anaconda3/envs/splendor_event311/bin/python')
    if os.path.isfile(known_good):
        return known_good

    shared = os.environ.get('SPLENDOR_INFERENCE_PYTHON')
    if shared and os.path.isfile(shared) and _supports_float_schedule(shared):
        return shared

    candidates = [
        os.path.expanduser('~/miniconda3/envs/splendor_event311/bin/python'),
        _find_inference_python(),
    ]

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
    _install_sb3_compat_shims()
    custom_objects = _build_sb3_custom_objects(obs_size=obs_size, action_space_size=action_space_size)
    if model_kind == 'maskable':
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(model_path, custom_objects=custom_objects)

    from stable_baselines3 import PPO
    return PPO.load(model_path, custom_objects=custom_objects)


def _resolve_value_model_info():
    """Resolve the score-based PPO model to use for the web value agent.

    Preference order:
    1. Explicit VALUE_AGENT_MODEL_PATH override (with optional VALUE_AGENT_MODEL_KIND)
    2. Default v3 maskable score model artifact
    3. Model artifact matching project/configs/training/maskable_ppo_v4a_ent_lr.yaml
    4. Legacy PPO v1 final model as final fallback
    """
    root = Path(project_root)
    config_path = os.path.join(project_root, 'project', 'configs', 'training', 'maskable_ppo_v4a_ent_lr.yaml')

    explicit_model_path = os.environ.get('VALUE_AGENT_MODEL_PATH')
    if explicit_model_path:
        explicit = Path(explicit_model_path).expanduser()
        if explicit.exists():
            env_kind = (os.environ.get('VALUE_AGENT_MODEL_KIND') or '').strip().lower()
            if env_kind not in ('maskable', 'ppo'):
                env_kind = 'maskable' if 'maskable' in explicit.name.lower() or 'maskable' in str(explicit).lower() else 'ppo'
            return {
                'model_path': str(explicit),
                'model_kind': env_kind,
                'model_source': 'env_VALUE_AGENT_MODEL_PATH',
            }
        print(f'Warning: VALUE_AGENT_MODEL_PATH does not exist: {explicit}')

    v3_default = root / 'project' / 'logs' / 'maskable_ppo_score_v3_20260303_183435' / 'final_model.zip'
    if v3_default.exists():
        return {
            'model_path': str(v3_default),
            'model_kind': 'maskable',
            'model_source': 'default_maskable_ppo_score_v3_20260303_183435',
        }

    raise FileNotFoundError(
        'No score-based PPO model found. Set VALUE_AGENT_MODEL_PATH to a valid model zip '
        'or ensure project/logs/maskable_ppo_score_v3_20260303_183435/final_model.zip exists.'
    )


class ScoreBasedPPOOpponent:
    """Web opponent adapter for trained score-based PPO / MaskablePPO models."""

    def __init__(self):
        model_info = _resolve_value_model_info()
        self.model_path = model_info['model_path']
        self.model_kind = model_info['model_kind']
        self.model_source = model_info['model_source']
        self.vectorizer = SplendorStateVectorizer()
        self.inference_script = os.path.join(script_dir, 'web_score_inference.py')
        self.python_cmd = _find_value_inference_python()
        self.strict_inference = True
        print(f'ScoreBasedPPOOpponent: model={self.model_path} python={self.python_cmd}')

    def choose_web_action(self, game_env_obj):
        game_env_obj.update_actions()
        legal_actions = game_env_obj.action_space.list_of_actions
        if not legal_actions:
            return None

        state = game_env_obj.current_state_of_the_game
        obs = self.vectorizer.vectorize(state, player_id=state.active_player_id, turn_count=0)

        payload = {
            'model_path': self.model_path,
            'model_kind': self.model_kind,
            'observation': obs.tolist(),
            'legal_actions_count': len(legal_actions),
        }

        try:
            env = os.environ.copy()
            env.setdefault('LANG', 'en_US.UTF-8')
            env.setdefault('LC_ALL', 'en_US.UTF-8')
            env.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
            completed = subprocess.run(
                [self.python_cmd, self.inference_script],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=True,
                env=env,
                timeout=120,
            )
            result = json.loads(completed.stdout)
            action_idx = int(result.get('action_idx', 0))
        except Exception as exc:
            raise RuntimeError(f'score PPO inference failed: {exc}') from exc

        if action_idx < 0 or action_idx >= len(legal_actions):
            raise RuntimeError(f'value model returned invalid action index {action_idx} for {len(legal_actions)} legal actions')

        return legal_actions[action_idx]





def _resolve_event_model_info():
    """Use the ppo_event_based_A_hybrid_start_s42_20260328_155415_1000000_steps.zip model for the event-based opponent.

    This artifact is loaded via PPO (non-maskable) for compatibility.
    """
    root = Path(project_root)
    model_path = root / 'project_event_based' / 'notebooks' / 'models' / 'A_hybrid_start' / 'ppo_event_based_A_hybrid_start_s42_20260328_155415_1000000_steps.zip'
    if not model_path.exists():
        raise FileNotFoundError(f'Event model not found: {model_path}')
    return {
        'model_path': str(model_path),
        'model_kind': 'maskable',
        'model_source': 'ppo_event_based_A_hybrid_start_s42_20260328_155415_1000000_steps',
    }


class EventBasedPPOOpponent:
    """Web opponent using the trained event-based PPO model via subprocess inference."""

    def __init__(self):
        model_info = _resolve_event_model_info()
        self.model_path = model_info['model_path']
        self.model_kind = model_info['model_kind']
        self.inference_script = os.path.join(script_dir, 'web_score_inference.py')
        self.python_cmd = _find_event_inference_python(self.model_path)
        self.infer_timeout_sec = float(os.environ.get('EVENT_INFERENCE_TIMEOUT_SEC', '12'))
        self.first_infer_timeout_sec = float(os.environ.get('EVENT_FIRST_INFERENCE_TIMEOUT_SEC', '45'))
        self._worker_warmed = False
        self._first_move_pending = True
        # Keep the same event-history signal used during training.
        self.last_event = np.zeros(9, dtype=np.float32)
        self.infer_client = _PersistentInferenceClient(
            python_cmd=self.python_cmd,
            inference_script=self.inference_script,
            timeout_sec=self.infer_timeout_sec,
        )
        self.cached_model = None
        if os.environ.get('WEB_INPROCESS_FASTPATH', '').lower() in ('1', 'true', 'yes'):
            self._init_cached_model()
        # Warm up in background so first AI turn is less likely to timeout.
        #self._warm_up_worker()
        print(f'EventBasedPPOOpponent: model={self.model_path}')

    def reset_context(self):
        """Reset per-game recurrent context (event history tail)."""
        self.last_event = np.zeros(9, dtype=np.float32)
        self._first_move_pending = True

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
                self._worker_warmed = True
                print('EventBasedPPOOpponent: persistent inference worker warmed up.')
            except Exception as exc:
                warm_client.close()
                print(f'Warning: event worker warmup failed ({exc}); will keep lazy init on first request.')
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
        """135-d raw obs aligned with project_event_based training (40-d event prefix + zero pad)."""
        try:
            return _state_to_event_training_raw135(state, int(state.active_player_id))
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

    def _update_last_event_round(self, state_before_human, human_action, state_after_ai):
        """One event tail per human+AI round, matching train_maskable + SplendorGymWrapper.

        Training wraps one learner step that runs the learner action then the opponent; the returned
        135-d obs is after both. We mirror that with state_before_human (human to act), human_action,
        and state_after_ai (after AI replied). prev/next 40-d slices use learner perspective (player 0),
        matching EventRewardWrapper + fixed player_id in SplendorGymWrapper._get_observation.
        """
        if _event_detect is None:
            self.last_event = np.zeros(9, dtype=np.float32)
            return

        learner_id = 0
        prev_vec = _state_to_event_training_raw135(state_before_human, learner_id)[:40]
        next_vec = _state_to_event_training_raw135(state_after_ai, learner_id)[:40]

        if np.array_equal(prev_vec, next_vec):
            self.last_event = np.zeros(9, dtype=np.float32)
            return

        try:
            ev = _event_detect(prev_vec, human_action, next_vec)
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
                self._first_move_pending = False
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
            timeout_sec = 0.0 if self._first_move_pending else self.infer_timeout_sec
            action_idx = self.infer_client.predict_action_idx(payload, timeout_sec=timeout_sec)
            if 0 <= action_idx < len(legal_actions):
                self._worker_warmed = True
                chosen = legal_actions[action_idx]
                self._first_move_pending = False
                return chosen
        except Exception as exc:
            raise RuntimeError(f'event PPO inference failed: {exc}') from exc


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

    history_file = os.path.join(script_dir, 'outputs', 'web_game_history.jsonl')
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
                raise RuntimeError(f'could not load {opponent_type} PPO model: {exc}') from exc
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

    try:
        state = init_game(selected_opponent_type)
    except Exception as exc:
        return jsonify({'error': f'Failed to initialize game: {exc}'}), 500
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

    state_before_human = copy.deepcopy(game_env.current_state_of_the_game)
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
        try:
            timeout_sec = _resolve_ai_action_timeout(ai_agent)
            ai_action = _choose_ai_action_with_timeout(ai_agent, game_env, legal_actions, timeout_sec=timeout_sec)
        except Exception as exc:
            return jsonify({'error': f'AI model inference failed: {exc}'}), 500
    else:
        # Random opponent
        import random
        ai_action = random.choice(legal_actions) if legal_actions else None

    state_after_ai = None
    if ai_action is None:
        # No action possible
        game_done = True
        winner = 'human'
    else:
        obs, reward, done, info = game_env.step('deterministic', ai_action, return_observation=False)
        state_after_ai = game_env.current_state_of_the_game
        # Update legal actions after AI move
        game_env.update_actions()
        _update_final_round_status()

    if opponent_type == 'event' and ai_agent is not None and state_after_ai is not None:
        try:
            if hasattr(ai_agent, '_update_last_event_round'):
                ai_agent._update_last_event_round(state_before_human, action, state_after_ai)
        except Exception:
            pass

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

    app.run(debug=True, host='0.0.0.0', port=8081, use_reloader=False)