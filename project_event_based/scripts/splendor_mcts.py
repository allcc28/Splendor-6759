import numpy as np
import torch as th

class SplendorNode:
    def __init__(self, state_bytes, obs, action_mask, parent=None, action_taken=None, prior_prob=0.0):
        self.state_bytes = state_bytes  # the "snapshot" of the game state at this node, stored as bytes for fast copying
        self.obs = obs                  # the observation fed to the PPO agent
        self.action_mask = action_mask  # the mask of legal actions (crucial for preventing illegal moves in MCTS)
        
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.expanded = False
        
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior_prob
        
        # pre-allocate arrays for child statistics to avoid dynamic resizing during search
        self.child_priors = np.zeros(200, dtype=np.float32)
        self.child_visits = np.zeros(200, dtype=np.float32)
        self.child_values = np.zeros(200, dtype=np.float32)

    def expand(self, action_probs):
        self.expanded = True
        self.child_priors = action_probs

    def get_best_action_puct(self, cpuct=1.5):
        # vectorized PUCT calculation for all children
        q_values = np.divide(self.child_values, self.child_visits, out=np.zeros_like(self.child_values), where=self.child_visits!=0)
        u_values = cpuct * self.child_priors * np.sqrt(self.visits) / (1 + self.child_visits)
        
        puct_scores = q_values + u_values
        
        # 【core action mask】set the scores of illegal actions to negative infinity, MCTS will never choose them
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
    def __init__(self, model, num_simulations=100, cpuct=1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.cpuct = cpuct

    def get_policy_and_value(self, obs, action_mask):
        """Brain-computer interface: bypass predict, directly extract P and V from SB3's underlying network"""
        obs_tensor = th.as_tensor(obs).unsqueeze(0).float()
        with th.no_grad():
            features = self.model.policy.mlp_extractor(obs_tensor)
            
            # Extract Actor network (policy P)
            logits = self.model.policy.action_net(features.latent_pi)
            logits[0, ~th.tensor(action_mask)] = -1e9 # Mask 掉非法动作的 logits
            probs = th.softmax(logits, dim=-1)[0].numpy()
            
            # Extract Critic network (value V)
            value = self.model.policy.value_net(features.latent_vf)[0].item()
            
        return probs, value

    def search(self, env):
        # 1. save the original state of the environment to allow perfect restoration after simulations
        original_state = env.unwrapped.get_state()
        initial_obs = env.unwrapped._get_observation()
        initial_mask = env.unwrapped.get_action_mask()
        
        root = SplendorNode(state_bytes=original_state, obs=initial_obs, action_mask=initial_mask)

        for _ in range(self.num_simulations):
            node = root
            
            # 2. before each simulation, let the environment rewind to the root node
            env.unwrapped.set_state(root.state_bytes)
            
            is_terminal = False
            agent_won = False
            
            # Phase 1: tree search (Selection)
            while node.expanded and not is_terminal:
                action = node.get_best_action_puct(self.cpuct)
                
                # interact with the environment using the selected action, observe the outcome
                obs, reward, terminated, truncated, info = env.step(action)
                is_terminal = terminated or truncated
                
                if action not in node.children:
                    # This is the first time we visit this child node, we need to create it and evaluate it with the neural network
                    new_state = env.unwrapped.get_state()
                    new_mask = env.unwrapped.get_action_mask()
                    
                    node.children[action] = SplendorNode(
                        state_bytes=new_state, 
                        obs=obs, 
                        action_mask=new_mask,
                        parent=node, 
                        action_taken=action
                    )
                    agent_won = info.get('agent_won', False)
                
                node = node.children[action]

            # Phase 2: (Evaluation)
            if not is_terminal:
                probs, value = self.get_policy_and_value(node.obs, node.action_mask)
                node.expand(probs)
            else:
                value = 1.0 if agent_won else -1.0 

            # Phase 3: (Backpropagation)
            node.backpropagate(value)

        # 3. after simulations, restore the environment to the original state before returning the best action
        env.unwrapped.set_state(original_state)

        #return the action with the most visits (i.e., the highest win rate and the most thoroughly validated)
        return int(np.argmax(root.child_visits))