import logging
from gym_open_ai.envs.registration import register, error

logger = logging.getLogger(__name__)

# Register the Splendor environments if they are not already registered.
# This guards against multiple imports in the same session (e.g., during interactive use).
for env_id, entry in [
    ('splendor-v0', 'gym_splendor_code.envs.splendor:SplendorEnv'),
    ('splendor-v1', 'gym_splendor_code.envs.splendor_wrapper:SplendorWrapperEnv'),
    ('splendor-deterministic-v0', 'gym_splendor_code.envs.splendor_deterministic:SplendorDeterministic'),
]:
    try:
        register(id=env_id, entry_point=entry)
    except error.Error:
        # Already registered, ignore.
        logger.debug(f"Environment '{env_id}' is already registered.")
