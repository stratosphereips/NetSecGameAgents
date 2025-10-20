from AIDojoCoordinator.game_components import Action, Observation, AgentStatus
from agents.base_agent import BaseAgent
# from agents.action_list_base_agent import BaseAgent
from agents.agent_utils import generate_valid_actions, state_as_ordered_string, filter_log_files_from_state
from random import choice

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import json
from collections import deque
# import mlflow
from os import path, makedirs
import argparse
import logging

import wandb

class EpsilonScheduler:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.05, decay_rate=1e-4):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.step_count = 0

    def get_epsilon(self):
        # Exponential decay formula
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-self.decay_rate * self.step_count)
        return epsilon

    def step(self):
        self.step_count += 1

class TextEncoder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
    # def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model.eval()  # Set the model to evaluation mode

    def encode(self, input_data):
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data, sort_keys=True)
        elif isinstance(input_data, str):
            input_str = input_data
        else:
            raise ValueError("Unsupported input type")
        with torch.no_grad():
            embedding = self.model.encode(input_str)
        return torch.tensor(embedding).float()
    

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)  # Output is a single Q-value

    def forward(self, state_emb, action_emb):
        # x = torch.cat((state_emb, dim=-1)
        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)
            action_emb = action_emb.unsqueeze(0)
        # Concatenate the embeddings along the feature dimension (dim=1)
        x = torch.cat([state_emb, action_emb], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DualReplayBuffer:
    def __init__(self, capacity=10000):
        self.success = deque(maxlen=capacity // 2)
        self.general = deque(maxlen=capacity)

    def push(self, state, action_emb, reward, next_state, done):
        self.general.append((state, action_emb, reward, next_state, done))
        if reward > 0:
            # print("Storing in success buffer")
            # print(f"Action: {action}, Reward: {reward}, Next state: {next_state}, Done: {done}")
            self.success.append((state, action_emb, reward, next_state, done))

    def sample(self, batch_size):
        half = batch_size // 2
        sample_general = random.sample(self.general, min(half, len(self.general)))
        sample_success = random.sample(
            self.success, min(batch_size - len(sample_general), len(self.success))
        )
        return sample_general + sample_success

    def __len__(self):
        return len(self.general) + len(self.success)


class DDQNAgent(BaseAgent):

    def __init__(self, host, port, role, epsilon_decay=1e-4, lr=0.01, gamma=0.99):
        super().__init__(host, port, role)

        self.state_encoder = TextEncoder()
        self.action_encoder = TextEncoder(model_name="all-MiniLM-L6-v2")       
        
        self.gamma = gamma
        self.buffer = DualReplayBuffer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_set = {}
        self.action_set = {}
        self.lr = lr
        self.epsilon_end: float = 0.01
        self.epsilon_decay: float = epsilon_decay
        self.epsilon:float = 1.0
        self.max_win_rate = 0.0

    def define_networks(self):
        self.q_net = QNetwork(self.state_encoder.embedding_dim, self.action_encoder.embedding_dim)
        self.target_net = QNetwork(self.state_encoder.embedding_dim, self.action_encoder.embedding_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.q_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    @torch.no_grad()
    def select_action(self, observation: Observation, epsilon: float):
        # return the index of the best action based on the Q-values
        # state_str = str(observation.state)
        state_str = state_as_ordered_string(observation.state)
        if state_str not in self.state_set.keys():
            state_emb = self.state_encoder.encode(state_str).to(self.device)
            self.state_set[state_str] = state_emb
        else:
            state_emb = self.state_set[state_str].to(self.device)
        
        valid_actions = generate_valid_actions(observation.state)
        if random.random() < epsilon:
            action = choice(valid_actions)
            return action, self.action_encoder.encode(str(action)).to(self.device)

        valid_action_embs = []
        for action in valid_actions:
            if str(action) not in self.action_set.keys():
                action_emb = self.action_encoder.encode(str(action)).to(self.device)
                self.action_set[str(action)] = action_emb
                valid_action_embs.append(action_emb)
            else:
                valid_action_embs.append(self.action_set[str(action)].to(self.device))

        valid_action_embs = torch.stack(valid_action_embs)  # (N_valid, D_A)
        num_valid_actions = len(valid_action_embs)
    
        # 1. Expand state embedding to match the number of valid actions
        # (1, D_S) -> (N_valid, D_S)
        state_emb_expanded = state_emb.repeat(num_valid_actions, 1)
    
        # 2. Compute Q-values for all (s, a_valid) pairs
        # The net will take (N_valid, D_S) and (N_valid, D_A)
        q_values = self.q_net(state_emb_expanded, valid_action_embs) # (N_valid,)
    
        # 3. Find the action with the maximum Q-value
        best_action_idx = q_values.argmax()
    
        # 4. Return the corresponding action embedding
        best_action_emb = valid_action_embs[best_action_idx]
    
        return valid_actions[best_action_idx], best_action_emb
        

    def store(self, *args):
        self.buffer.push(*args)

    def update(self, episode:int, batch_size:int=32, num_epochs:int=1):
        if len(self.buffer) < batch_size:
            return

        for _ in range(num_epochs):
            batch = self.buffer.sample(batch_size)
            # print("Sampled batch size:", len(batch))

            # losses = []
            state_embs = []
            next_state_embs = []
            rewards = []
            dones = []
            action_embs_taken = []

            for observation, action_emb_taken, reward, next_observation, done in batch:

                state_str = state_as_ordered_string(observation.state)
                # state_str = str(observation)

                # Get the valid action mask for the current observation
                # valid_action_mask = self.get_valid_action_mask(observation)
                if state_str not in self.state_set.keys():
                    s = self.state_encoder.encode(state_str).to(self.device)
                    self.state_set[state_str] = s
                else:
                    s = self.state_set[state_str].to(self.device)
                
                rewards.append(reward)
                state_embs.append(s)
                action_embs_taken.append(action_emb_taken) # Action embedding of the action taken
                dones.append(done)


                with torch.no_grad():
                    next_state_str = state_as_ordered_string(next_observation.state)
                    # next_state_str = str(next_observation)
                    if next_state_str not in self.state_set.keys():
                        next_s = self.state_encoder.encode(next_state_str).to(self.device)
                        self.state_set[next_state_str] = next_s
                    else:
                        next_s =  s = self.state_set[next_state_str].to(self.device)
                next_state_embs.append(next_s)
                


            # Stack and compute batch loss
            state_batch = torch.stack(state_embs).to(self.device)
            next_state_batch = torch.stack(next_state_embs).to(self.device)
            actions_emb_taken_batch = torch.stack(action_embs_taken).to(self.device) # (B, D_A)
            dones = torch.tensor(dones, dtype=torch.float, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
            
            # --- Compute current Q(s,a) ---
            # Q-value for the state and the action actually taken
            q_values = self.q_net(state_batch, actions_emb_taken_batch) # (B,)
            q_values = q_values.squeeze(-1)
            
            # ---- Double DQN target Q ----
            with torch.no_grad():
                # --- Double DQN Target Q ---
                # This requires iterating through the batch since the number of valid actions varies.

                next_q_values = []
                # next_actions = generate_valid_actions(next_observation)

                for i in range(len(next_state_batch)):


                    # State embeddings for the i-th next state (1, D_S)
                    next_s_emb = next_state_batch[i].unsqueeze(0) 

                    # 1. Online net selects the best next action (action a')
                    _, best_action_emb_online = self.select_action(next_observation, 0.0) # (D_A)

                    # 2. Target net evaluates Q(s', a')
                    # Use the target network to evaluate the action selected by the online network
                    next_q_value = self.target_net(
                        next_s_emb, 
                        best_action_emb_online.unsqueeze(0) # Need (1, D_A)
                    ).squeeze(0) # scalar

                    next_q_values.append(next_q_value)

                next_q_values = torch.tensor(next_q_values, dtype=torch.float, device=self.device) # (B,)

                # 3. Bellman target
                expected_q = rewards + self.gamma * next_q_values * (1 - dones)

            # loss = F.mse_loss(q_values, expected_q.detach())
            loss = F.smooth_l1_loss(q_values, expected_q.detach())


            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()
        # losses.append(loss.item())
        wandb.log({
            "loss": loss.item(),
            "episode": episode,
            "batch_size": batch_size,
        })
        
        # self.save()

    def update_target(self):
        self._logger.debug(f"Updating target network.")
        self.target_net.load_state_dict(self.q_net.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, online_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self, observation, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """

        epsilon_scheduler = EpsilonScheduler(epsilon_start=1.0, epsilon_end=self.epsilon_end, decay_rate=self.epsilon_decay)
        returns = []
        wins = 0
        observation = filter_log_files_from_state(observation)

        for ep in range(num_episodes):
            num_steps = 0
            self._logger.info(f"Playing episode {ep}")
            episodic_returns = []
            while observation and not observation.end:
                num_steps += 1
                self._logger.debug(f"Observation received:{observation}")
                # Store returns in the episode
                episodic_returns.append(observation.reward)
                # Select the best action
                # TODO: use epsilon decay starting with 1.0 and decaying to 0.01
                # self.epsilon = 1.0 - (ep / num_episodes) * 0.99
                # self.epsilon = max(self.epsilon, 0.01)
                self.epsilon = epsilon_scheduler.get_epsilon()
                action, action_emb = self.select_action(observation, epsilon=self.epsilon)

                next_observation = self.make_step(action)
                self._logger.debug(f"Before filtering: {next_observation.state}")
                next_observation = filter_log_files_from_state(next_observation)
                self._logger.debug(f"After filtering: {next_observation.state}")

                # Add intrinsic reward
                reward = next_observation.reward
                if observation != next_observation:
                     reward += 50
                else:
                    reward -= 50  

                reward = reward / 10.0  # Scale down the reward

                # print(f"Action: {self.get_action(action_id)}, Reward: {next_observation.reward}, Intrinsic reward: {reward - next_observation.reward}")

                self.store(
                    observation,
                    action_emb,
                    reward,
                    next_observation,
                    float(next_observation.end),
                )
                observation = next_observation
                # possible_actions = next_actions
                # Update the network after each step
                self.update(episode=ep)
                self.soft_update(tau=0.005)
                # self.update_epsilon()

        
            # Add the last observation reward to the episodic returns
            if observation.info and observation.info["end_reason"] == AgentStatus.Success:
                wins += 1
                self._logger.info(f"Episode {ep} ended with success.")
                # Make sure the last one is added
                episodic_returns.append(observation.reward)

            # if ep % 12 == 0 and ep != 0:
            #     self.update_target()

            # Update the epsilon counter after each episode
            epsilon_scheduler.step()

            if ep % 2000 == 0 and ep != 0:
                self.save(f"checkpoints/ddqn_checkpoint_{ep}.pt")

            if ep % 200 == 0 and ep != 0:
                # Run evaluation every 200 episodes
                self.q_net.eval()
                win_rate = self.eval(self.request_game_reset(), ep, num_episodes=250)
                self.q_net.train()
                if win_rate > self.max_win_rate:
                    self.max_win_rate = win_rate
                    self.save("checkpoints/ddqn_checkpoint_best.pt")

                if win_rate >= 98.0:
                    self._logger.info(f"Early stopping at episode {ep} with win rate {win_rate}%")
                    print(f"Early stopping at episode {ep} with win rate {win_rate}%")
                    break

            self._logger.debug(
                    f"Episode: {ep} ReplayBuffer: {len(self.buffer.general)} general, {len(self.buffer.success)} success"
                )
            # print(f"Episode: {ep} ReplayBuffer: {len(self.buffer.general)} general, {len(self.buffer.success)} success")

            # To return
            # last_observation = observation
            self._logger.debug(f"Observation received:{observation}")
            returns.append(np.sum(episodic_returns))
            self._logger.info(
                f"Episode {ep} ended with return {np.sum(episodic_returns)} in {num_steps} steps. Mean returns={np.mean(returns)}±{np.std(returns)}"
            )
            
            print(f"Episode {ep} ended with return {np.sum(episodic_returns)} in {num_steps} steps. Mean returns={np.mean(returns)}±{np.std(returns)}")

            wandb.log({
                "episode": ep,
                "return": np.sum(episodic_returns),
                "mean_return": np.mean(returns),
                "std_return": np.std(returns),
                "wins": wins,
                "epsilon": self.epsilon,
                "win rate": (wins / (ep + 1)) * 100
            })
            # Reset the episode
            observation = self.request_game_reset()

        self._logger.info(
            f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}"
        )
        # Return the episode in case of early stopping
        return ep

    def eval(self, observation, train_episode, num_episodes=1) -> float:
        returns = []
        steps = []
        wins = 0
        for ep in range(num_episodes):
            num_steps = 0
            self._logger.info(f"Playing eval episode {ep}")
            episodic_returns = []
            while observation and not observation.end:
                num_steps += 1
                self._logger.debug(f"Observation received:{observation}")
                # Store returns in the episode
                episodic_returns.append(observation.reward)

                # Select the best action
                observation = filter_log_files_from_state(observation)
                (action, _) = self.select_action(observation, epsilon=0.0)
                # print(f"Selected action id: {self.get_action(action_id)}")
                next_observation = self.make_step(action)
                next_observation = filter_log_files_from_state(next_observation)

                observation = next_observation
                self._logger.debug(f"Observation received: {observation}")
            
            if observation.info and observation.info["end_reason"] == AgentStatus.Success:
                wins += 1
                self._logger.info(f"Episode {ep} ended with success.")
                # Make sure the last one is added
                episodic_returns.append(observation.reward)

            
            returns.append(np.sum(episodic_returns))
            steps.append(num_steps)

            self._logger.info(
                f"Eval episode {ep} ended with return {np.sum(episodic_returns)} in {num_steps} steps. Mean returns={np.mean(returns)}±{np.std(returns)}"
            )
            
            print( f"Eval episode {ep} ended with return {np.sum(episodic_returns)} in {num_steps} steps. Mean returns={np.mean(returns)}±{np.std(returns)}")

            # Reset the episode
            observation = self.request_game_reset()
            

        # self._logger.info(
        #     f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}"
        # )

        text = f"""Final results for {self.__class__.__name__} after {num_episodes} episodes: 
                Returns: {np.mean(returns)}±{np.std(returns)}
                Wins: {wins}
                Win rate: {(wins / num_episodes) * 100}%,
                Avg steps: {np.mean(steps)}
                """
        wandb.log({
            "episode": train_episode,
            "eval_mean_return": np.mean(returns),
            "eval_std_return": np.std(returns),
            "eval_wins": wins,
            "eval_win_rate": (wins / (ep + 1)) * 100,
            "eval_avg_steps": np.mean(steps),
            "eval_std_steps": np.std(steps)
        })
        
        print(text)
        return 100 * (wins/num_episodes)

    def save(self, path="ddqn_checkpoint.pt"):
        torch.save(
            {
                "q_net_state_dict": self.q_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        self._logger.debug(f"Model saved to {path}")

    def load(self, path="ddqn_checkpoint.pt"):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {path}")


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action="store", required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action="store", required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to play or evaluate", default=100, type=int)
    parser.add_argument("--logdir", help="Folder to store logs",default=path.join(path.dirname(path.abspath(__file__)), "logs"))
    parser.add_argument("--evaluate", help="Evaluate the agent and report, instead of playing the game only once.",action="store_true")
    # parser.add_argument("--cont", help="Continue training the final model from the previous run.", action="store_true")
    parser.add_argument("--env_conf", help="Configuration file of the env. Only for logging purposes.", required=False, default='./env/netsecenv_conf.yaml', type=str)
    parser.add_argument("--decay", help="Epsilon decay factor", required=False, default=1e-4, type=float)
    parser.add_argument("--lr", help="Learning rate", required=False, default=1e-3, type=float)

    args = parser.parse_args()

    if not path.exists(args.logdir):
        makedirs(args.logdir)
    logging.basicConfig(
        filename=path.join(args.logdir, "ddqn_agent.log"),
        filemode="w",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.ERROR,
    )

    wandb.init(
        project="UTEP-Collaboration",
        entity="stratosphere",
        name="DDQN-embed-black-box",
        config={
            "learning_rate": args.lr,
            "gamma": 0.99,
            "episodes": 50000,
            "batch_size": 32,
            "epsilon_decay_factor": args.decay,  # Decay factor for epsilon
        },
    )


    if path.exists(args.env_conf):
        wandb.save(args.env_conf, base_path=path.dirname(path.abspath(args.env_conf)))

    # Create agent
    agent = DDQNAgent(args.host, args.port, "Attacker", lr=args.lr, epsilon_decay=args.decay)
    

    if not args.evaluate:
        # Play the normal game
        observation = agent.register()
        # action_list = agent._action_list
        agent.define_networks()
        wandb.watch(agent.q_net, log="all")

        # if args.cont:
        #     print("Continuing training from the last checkpoint")
        #     agent.load("checkpoints/ddqn_checkpoint_final.pt")
        #     agent.q_net.train()
        #     agent.epsilon = 0.1  # Start with a lower epsilon for continued training

        num_episodes = agent.train(observation, args.episodes)
        agent.save("checkpoints/ddqn_checkpoint_final.pt")

        # Final evaluation
        agent.q_net.eval()
        agent.eval(observation, num_episodes, num_episodes=250)

        agent._logger.info("Terminating interaction")
        agent.terminate_connection()
        wandb.finish()
        
    else:
        # Evaluate the agent performance
        print("Evaluating the agent performance")

    #     # How it works:
    #     # - Evaluate for several 'episodes' (parameter)
    #     # - Each episode finishes with: steps played, return, win/lose. Store all
    #     # - Each episode compute the avg and std of all.
    #     # - Every X episodes (parameter), report in log and mlflow
    #     # - At the end, report in log and mlflow and console
        observation = agent.register()
        action_list = agent._action_list
        agent.define_networks(len(action_list))

    #     # Wandb experiment name
        experiment_name = "DDQN Embed Agent Eval"

        # Load the final checkpoint
        agent.load("checkpoints/ddqn_checkpoint_best.pt")
        agent.q_net.eval()

        agent.eval(observation, args.episodes, args.episodes)
        agent._logger.info("Terminating interaction")
        agent.terminate_connection()