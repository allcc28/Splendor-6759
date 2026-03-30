import numpy as np
import torch as th
import copy

class SplendorNode:
    def __init__(self, snapshot, obs, action_mask, parent=None, action_taken=None, prior_prob=0.0):
        self.snapshot = snapshot
        self.obs = obs
        self.action_mask = action_mask
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.expanded = False
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior_prob
        self.child_priors = np.zeros(200, dtype=np.float32)
        self.child_visits = np.zeros(200, dtype=np.float32)
        self.child_values = np.zeros(200, dtype=np.float32)

    def expand(self, action_probs):
        self.expanded = True
        self.child_priors = action_probs

    def get_best_action_puct(self, cpuct=2.0): 
        q_values = np.divide(self.child_values, self.child_visits, out=np.zeros_like(self.child_values), where=self.child_visits!=0)
        
        valid_q = q_values[self.action_mask & (self.child_visits > 0)]
        if len(valid_q) > 1 and np.max(valid_q) > np.min(valid_q):
            q_min = np.min(valid_q)
            q_max = np.max(valid_q)
            norm_q = (q_values - q_min) / (q_max - q_min)
        else:
            norm_q = np.zeros_like(q_values) 

        u_values = cpuct * self.child_priors * np.sqrt(self.visits + 1e-8) / (1 + self.child_visits)
        
        puct_scores = norm_q + u_values
        puct_scores[~self.action_mask] = -np.inf 
        
        return int(np.argmax(puct_scores))

    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.child_visits[self.action_taken] += 1
            self.parent.child_values[self.action_taken] += value
            self.parent.backpropagate(value)

class SplendorMCTS:
    def __init__(self, model, num_simulations=50, cpuct=1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.cpuct = cpuct

    def _save_state(self, env):
        unwrapped = env.unwrapped
        return {
            'core_state': copy.deepcopy(unwrapped.env.current_state_of_the_game),
            'turn_count': unwrapped.turn_count,
            'prev_score': unwrapped.prev_score,
            'last_event': getattr(env, 'last_event', np.zeros(9)).copy(),
            'last_obs_raw': getattr(env, 'last_obs_raw', np.zeros(135)).copy()
        }

    def _load_state(self, env, snapshot):
        unwrapped = env.unwrapped
        unwrapped.env.current_state_of_the_game = copy.deepcopy(snapshot['core_state'])
        unwrapped.turn_count = snapshot['turn_count']
        unwrapped.prev_score = snapshot['prev_score']
        unwrapped._update_legal_actions() 
        if hasattr(env, 'last_event'):
            env.last_event = snapshot['last_event'].copy()
            env.last_obs_raw = snapshot['last_obs_raw'].copy()

    def get_policy_and_value(self, obs, action_mask):
        obs_tensor = th.as_tensor(obs).unsqueeze(0).float()
        with th.no_grad():
            features = self.model.policy.mlp_extractor(obs_tensor)
            
            if isinstance(features, tuple):
                latent_pi, latent_vf = features
            else:
                latent_pi, latent_vf = features.latent_pi, features.latent_vf
            
            logits = self.model.policy.action_net(latent_pi)
            logits[0, ~th.tensor(action_mask)] = -1e9
            probs = th.softmax(logits, dim=-1)[0].numpy()
            
            value = self.model.policy.value_net(latent_vf)[0].item()
            
        return probs, value
            

    def search(self, env, current_obs):
        original_snapshot = self._save_state(env)
        initial_mask = env.action_masks()
        
        root = SplendorNode(snapshot=original_snapshot, obs=current_obs, action_mask=initial_mask)

        for _ in range(self.num_simulations):
            node = root
            self._load_state(env, root.snapshot)
            is_terminal = False
            agent_won = False
            
            while node.expanded and not is_terminal:
                action = node.get_best_action_puct(self.cpuct)
                obs, reward, terminated, truncated, info = env.step(action)
                is_terminal = terminated or truncated
                
                if action not in node.children:
                    new_snapshot = self._save_state(env)
                    new_mask = env.action_masks()
                    node.children[action] = SplendorNode(
                        snapshot=new_snapshot, obs=obs, action_mask=new_mask,
                        parent=node, action_taken=action
                    )
                    agent_won = info.get('agent_won', False)
                node = node.children[action]

            if not is_terminal:
                probs, value = self.get_policy_and_value(node.obs, node.action_mask)
                node.expand(probs)
            else:
                if info.get('agent_won', False):
                    value = 1000.0  
                elif info.get('agent_lost', False):
                    value = -1000.0
                else:
                    value = 0.0

            node.backpropagate(value)

        self._load_state(env, original_snapshot)
        return int(np.argmax(root.child_visits))