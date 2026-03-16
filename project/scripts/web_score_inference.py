#!/usr/bin/env python3
import json
import os
import sys
import time

# Must be set before numpy/torch load their OpenMP runtimes to avoid SIGABRT on macOS.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import importlib
import numpy as np

# -----------------------------------------------------------------------
# numpy._core compatibility shim for environments using numpy 1.x.
# Models pickled under numpy 2.x reference 'numpy._core.*'; this shim
# redirects those references to the numpy 1.x 'numpy.core.*' equivalents.
# -----------------------------------------------------------------------
sys.modules.setdefault('numpy._core', np.core)

_submodule_names = [
    'numeric', 'multiarray', 'umath', 'fromnumeric', 'function_base',
    '_ufunc_config', '_dtype', '_type_aliases', 'arrayprint', 'shape_base',
    '_methods', 'overrides', 'records', 'defchararray', 'numerictypes',
    'einsumfunc', 'getlimits', 'memmap', 'machar', '_add_newdocs',
    '_add_newdocs_scalars', '_exceptions', '_internal', '_string_helpers',
]
for _name in _submodule_names:
    _dst_key = f'numpy._core.{_name}'
    if _dst_key not in sys.modules:
        _src_key = f'numpy.core.{_name}'
        if _src_key in sys.modules:
            sys.modules[_dst_key] = sys.modules[_src_key]
        else:
            try:
                sys.modules[_dst_key] = importlib.import_module(_src_key)
            except ImportError:
                pass

# Ensure already-loaded numpy.core.* are reflected as numpy._core.*
for _key, _val in list(sys.modules.items()):
    if _key.startswith('numpy.core.') and 'numpy._core' + _key[len('numpy.core'):] not in sys.modules:
        sys.modules['numpy._core' + _key[len('numpy.core'):]] = _val

# -----------------------------------------------------------------------
# numpy.random BitGenerator compat shim.
# Models saved under numpy 2.x pass the bit-generator CLASS to
# __bit_generator_ctor, but numpy 1.x expects a STRING name.
# -----------------------------------------------------------------------
import numpy.random._pickle as _np_rand_pickle
_orig_bit_gen_ctor = _np_rand_pickle.__bit_generator_ctor

def _compat_bit_gen_ctor(bit_generator=None):
    if isinstance(bit_generator, type):
        return bit_generator()
    return _orig_bit_gen_ctor(bit_generator)

_np_rand_pickle.__bit_generator_ctor = _compat_bit_gen_ctor

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')


def _build_custom_objects(payload, obs_size):
    """Return SB3 custom_objects to bypass deserialization failures caused by
    SB3/gymnasium version mismatches between training and inference environments."""
    try:
        import gymnasium.spaces as gspaces
    except ImportError:
        import gym.spaces as gspaces
    act_size = int(payload.get('action_space_size', 200))
    return {
        'observation_space': gspaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32),
        'action_space': gspaces.Discrete(act_size),
        'clip_range': lambda _: 0.2,
        'lr_schedule': lambda _: 3e-4,
    }


def _load_model(payload, obs_size):
    """Load a PPO/MaskablePPO model with compatibility custom objects."""
    model_path = payload['model_path']
    model_kind = payload['model_kind']
    custom_objects = _build_custom_objects(payload, obs_size)

    if model_kind == 'maskable':
        from sb3_contrib import MaskablePPO
        try:
            return MaskablePPO.load(model_path)
        except Exception:
            return MaskablePPO.load(model_path, custom_objects=custom_objects)

    from stable_baselines3 import PPO
    try:
        return PPO.load(model_path)
    except Exception:
        return PPO.load(model_path, custom_objects=custom_objects)


def _predict_action(payload, model_cache=None):
    """Run one prediction with optional model cache for serve mode."""
    t0 = time.time()
    model_path = payload['model_path']
    model_kind = payload['model_kind']
    observation = np.asarray(payload['observation'], dtype=np.float32)
    legal_actions_count = int(payload['legal_actions_count'])
    obs_size = len(observation)

    cache_key = (model_path, model_kind, int(payload.get('action_space_size', 200)), obs_size)
    model = None
    if model_cache is not None:
        model = model_cache.get(cache_key)

    if model is None:
        t_load0 = time.time()
        model = _load_model(payload, obs_size)
        print(f'[infer] model_loaded kind={model_kind} sec={time.time()-t_load0:.3f}', file=sys.stderr, flush=True)
        if model_cache is not None:
            model_cache[cache_key] = model

    if model_kind == 'maskable':
        action_mask = np.zeros(200, dtype=bool)
        action_mask[:max(0, min(legal_actions_count, 200))] = True
        t_pred0 = time.time()
        action, _ = model.predict(observation, action_masks=action_mask, deterministic=True)
        print(f'[infer] predict_done maskable sec={time.time()-t_pred0:.3f}', file=sys.stderr, flush=True)
    else:
        t_pred0 = time.time()
        action, _ = model.predict(observation, deterministic=True)
        print(f'[infer] predict_done ppo sec={time.time()-t_pred0:.3f}', file=sys.stderr, flush=True)

    action_idx = int(np.asarray(action).squeeze())
    print(f'[infer] total_sec={time.time()-t0:.3f}', file=sys.stderr, flush=True)
    return {'action_idx': action_idx}


def serve_forever():
    """Persistent JSONL inference server over stdin/stdout.

    Uses readline() instead of 'for line in stdin' to avoid block-buffering
    deadlock: when stdout is a pipe Python buffers stdin reads in 8 KB chunks,
    which means the child waits for data that never arrives until the parent
    closes the pipe.  readline() returns as soon as it sees '\\n'.
    """
    import io
    # Wrap stdin in a line-buffered text stream so each newline-terminated
    # write from the parent is delivered immediately.
    in_stream = io.TextIOWrapper(sys.stdin.buffer, line_buffering=True)
    model_cache = {}
    while True:
        raw_line = in_stream.readline()
        if not raw_line:   # EOF – parent closed stdin
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            result = _predict_action(payload, model_cache=model_cache)
        except Exception as exc:
            result = {'error': str(exc)}
        sys.stdout.write(json.dumps(result) + '\n')
        sys.stdout.flush()


def main():
    payload = json.load(sys.stdin)
    result = _predict_action(payload, model_cache=None)
    print(json.dumps(result))


if __name__ == '__main__':
    if '--serve' in sys.argv:
        serve_forever()
    else:
        main()