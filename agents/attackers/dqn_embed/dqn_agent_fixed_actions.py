from AIDojoCoordinator.game_components import Action, Observation, AgentStatus
# from agents.base_agent import BaseAgent
from agents.action_list_base_agent import ActionListAgent
from agents.agent_utils import generate_valid_actions, state_as_ordered_string
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

class TextEncoder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
    # def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

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
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, action_dim)  # Output is a single Q-value

    def forward(self, state_emb):
        # x = torch.cat((state_emb, dim=-1)
        x = F.relu(self.fc1(state_emb))
        x = F.relu(self.fc2(x))
        return self.out(x)


# class ReplayBuffer:
#     def __init__(self, capacity=10000):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)

#     def __len__(self):
#         return len(self.buffer)


class DualReplayBuffer:
    def __init__(self, capacity=10000):
        self.success = deque(maxlen=capacity // 2)
        self.general = deque(maxlen=capacity)

    def push(self, state, valid_mask, action, reward, next_state, next_valid_mask, done):
        self.general.append((state, valid_mask, action, reward, next_state, next_valid_mask, done))
        if reward > 0:
            # print("Storing in success buffer")
            # print(f"Action: {action}, Reward: {reward}, Next state: {next_state}, Done: {done}")
            self.success.append((state, valid_mask, action, reward, next_state, next_valid_mask, done))

    def sample(self, batch_size):
        half = batch_size // 2
        sample_general = random.sample(self.general, min(half, len(self.general)))
        sample_success = random.sample(
            self.success, min(batch_size - len(sample_general), len(self.success))
        )
        return sample_general + sample_success

    def __len__(self):
        return len(self.general) + len(self.success)


class DQNAgent(ActionListAgent):

    def __init__(self, host, port, role, lr=0.001, gamma=0.99):
        super().__init__(host, port, role)

        self.encoder = TextEncoder()        
        
        self.gamma = gamma
        self.buffer = DualReplayBuffer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_set = {}
        self.action_set = {}
        self.lr = lr
        self.epsilon_end: float = 0.01
        self.epsilon_decay: float = 0.995
        self.epsilon:float = 1.0

    def define_networks(self, action_size:int):
        self.q_net = QNetwork(self.encoder.embedding_dim, action_size)
        self.target_net = QNetwork(self.encoder.embedding_dim, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.q_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def select_action(self, observation: Observation, epsilon: float) -> int:
        # return the index of the best action based on the Q-values
        state_str = state_as_ordered_string(observation.state)
        if state_str not in self.state_set.keys():
            state_emb = self.encoder.encode(state_str).to(self.device)
            self.state_set[state_str] = state_emb
        else:
            state_emb = self.state_set[state_str].to(self.device)
        
        # Start with random action with probability epsilon
        valid_action_mask = self.get_valid_action_mask(observation.state)
        # print(f"Mask len: {len(valid_action_mask)}, Mask sum: {sum(valid_action_mask)}")
        if random.random() < epsilon:
            # possible_actions = generate_valid_actions(observation.state)
            # Mask out invalid actions based on the valdi action maske
            valid_actions = [action for action, valid in zip(self._action_list, valid_action_mask) if valid]
            # print(f"Possible actions: {possible_actions}")
            random_action =  choice(valid_actions)
            # print(f"Random action selected: {random_action}")
            idx = self.get_action_index(random_action)
            # print(f"Random action index: {idx}")
            return idx
        
        q_values = []
        with torch.no_grad():

            q_values = self.q_net(state_emb.unsqueeze(0).to(self.device)).squeeze(0)
            # Mask out invalid actions based on the valid action mask
            q_values = q_values * torch.tensor(valid_action_mask, device=self.device)
        
        return torch.argmax(q_values).cpu().item()

    def store(self, *args):
        self.buffer.push(*args)

    def update(self, episode:int, batch_size:int=32, num_epochs:int=4):
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
            actions = []
            valid_action_masks = []
            next_valid_action_masks = []
            for observation, valid_mask, action_id, reward, next_observation, next_valid_mask, done in batch:
                # next_actions = generate_valid_actions(next_observation)

                state_str = state_as_ordered_string(observation)

                # Get the valid action mask for the current observation
                # valid_action_mask = self.get_valid_action_mask(observation)
                if state_str not in self.state_set.keys():
                    s = self.encoder.encode(state_str).to(self.device)
                    self.state_set[state_str] = s
                else:
                    s = self.state_set[state_str].to(self.device)
                
                rewards.append(reward)
                state_embs.append(s)
                actions.append(action_id)
                dones.append(done)
                valid_action_masks.append(valid_mask)
                next_valid_action_masks.append(next_valid_mask)

                # Compute target Q
                with torch.no_grad():
                    next_state_str = state_as_ordered_string(next_observation)
                    if next_state_str not in self.state_set.keys():
                        next_s = self.encoder.encode(next_state_str).to(self.device)
                        self.state_set[next_state_str] = next_s
                    else:
                        next_s =  s = self.state_set[next_state_str].to(self.device)
                next_state_embs.append(next_s)
                # next_valid_action_mask = self.get_valid_action_mask(next_observation)
                


            # Stack and compute batch loss
            state_batch = torch.stack(state_embs).to(self.device)
            next_state_batch = torch.stack(next_state_embs).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
            valid_action_mask_batch = torch.tensor(np.array(valid_action_masks), dtype=torch.float, device=self.device)
            next_valid_action_mask_batch = torch.tensor(np.array(next_valid_action_masks), dtype=torch.float, device=self.device)

            # Get q-values and then mask out invalid actions based on the valid action mask            
            q_values = self.q_net(state_batch)
            # print("q_values shape:", q_values.shape)
            q_values = q_values * valid_action_mask_batch
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                # Normal DQN
                next_q_values = self.target_net(next_state_batch)
                # print("next q_values shape:", next_q_values.shape)
                next_q_values = next_q_values * next_valid_action_mask_batch
                # print("next q_values shape after mask:", next_q_values.shape)
                
                next_q_values = next_q_values.max(1)[0]
                # print("next q_values shape: final:", next_q_values.shape)
                
                expected_q = rewards + self.gamma * next_q_values * (1 - dones)
                # DDQN
                # Action selection using main network
                # next_actions = self.q_net(next_state_batch).max(1)[1]
                # Action evaluation using target network
                # next_q_values = self.target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze()
                # expected_q = rewards + (self.gamma * next_q_values * ~dones)

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
    
    # def update_epsilon(self):
    #     """Decay epsilon"""
    #     self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self, observation, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        returns = []
        wins = 0
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
                self.epsilon = 1.0 - (ep / num_episodes) * 0.99
                self.epsilon = max(self.epsilon, 0.01)

                action_id = self.select_action(observation, epsilon=self.epsilon)
                next_observation = self.make_step(self.get_action(action_id))

                # Add intrinsic reward
                reward = next_observation.reward
                if observation != next_observation:
                     reward += 20
                else:
                    reward -= 10  

                # reward = max(min(reward, 1.0), -1.0)

                # print(f"Action: {self.get_action(action_id)}, Reward: {next_observation.reward}, Intrinsic reward: {reward - next_observation.reward}")

                self.store(
                    observation.state,
                    np.array(self.get_valid_action_mask(observation.state)),
                    action_id,
                    reward,
                    next_observation.state,
                    np.array(self.get_valid_action_mask(next_observation.state)),
                    float(next_observation.end),
                )
                observation = next_observation
                # possible_actions = next_actions
                # Update the network after each episode instead of each step
                self.update(episode=ep)
                # self.update_epsilon()

        
            # Add the last observation reward to the episodic returns
            if observation.info and observation.info["end_reason"] == AgentStatus.Success:
                wins += 1
                self._logger.info(f"Episode {ep} ended with success.")
                # Make sure the last one is added
                episodic_returns.append(observation.reward)

            if ep % 4 == 0:
                self.update_target()

            if ep % 500 == 0 and ep != 0:
                self.save(f"checkpoints/dqn_checkpoint_{ep}.pt")

            if ep % 200 == 0 and ep != 0:
                # Run evaluation every 200 episodes
                self.q_net.eval()
                win_rate = self.eval(self.request_game_reset(), ep, num_episodes=100)
                self.q_net.train()
                if win_rate >= 90.0:
                    self._logger.info(f"Early stopping at episode {ep} with win rate {win_rate}%")
                    print(f"Early stopping at episode {ep} with win rate {win_rate}%")
                    break

            self._logger.debug(
                    f"Episode: {ep} ReplayBuffer: {len(self.buffer.general)} general, {len(self.buffer.success)} success"
                )
            # print(f"Episode: {ep} ReplayBuffer: {len(self.buffer.general)} general, {len(self.buffer.success)} success")

            # To return
            last_observation = observation
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
        # This will be the last observation played before the reset
        # TODO: do we really need to return the last observation?
        return (last_observation, num_steps)

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
                action_id = self.select_action(observation, epsilon=0.0)
                # print(f"Selected action id: {self.get_action(action_id)}")
                next_observation = self.make_step(self.get_action(action_id))

                observation = next_observation
            
            if observation.info and observation.info["end_reason"] == AgentStatus.Success:
                wins += 1
                self._logger.info(f"Episode {ep} ended with success.")
                # Make sure the last one is added
                episodic_returns.append(observation.reward)

            self._logger.debug(f"Observation received:{observation}")
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

    def save(self, path="dqn_checkpoint.pt"):
        torch.save(
            {
                "q_net_state_dict": self.q_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        self._logger.debug(f"Model saved to {path}")

    def load(self, path="dqn_checkpoint.pt"):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {path}")


if __name__ == "__main__":

    wandb.init(
        project="UTEP-Collaboration",
        entity="stratosphere",
        name="DQN-embed-fixed-actions",
        config={
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "episodes": 10000,
            "batch_size": 32,
            "epsilon_decay_factor": 0.9,  # Decay factor for epsilon
        },
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        help="Host where the game server is",
        default="127.0.0.1",
        action="store",
        required=False,
    )
    parser.add_argument(
        "--port",
        help="Port where the game server is",
        default=9000,
        type=int,
        action="store",
        required=False,
    )
    parser.add_argument(
        "--episodes",
        help="Sets number of episodes to play or evaluate",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--test_each",
        help="Evaluate performance during testing every this number of episodes.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--logdir",
        help="Folder to store logs",
        default=path.join(path.dirname(path.abspath(__file__)), "logs"),
    )
    parser.add_argument(
        "--evaluate",
        help="Evaluate the agent and report, instead of playing the game only once.",
        action="store_true",
    )

    args = parser.parse_args()

    if not path.exists(args.logdir):
        makedirs(args.logdir)
    logging.basicConfig(
        filename=path.join(args.logdir, "dqn_agent.log"),
        filemode="w",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.WARNING,
    )

    # Create agent
    agent = DQNAgent(args.host, args.port, "Attacker")
    

    if not args.evaluate:
        # Play the normal game
        observation = agent.register()
        action_list = agent._action_list
        agent.define_networks(len(action_list))
        wandb.watch(agent.q_net, log="all")

        agent.train(observation, args.episodes)
        agent.save("checkpoints/dqn_checkpoint_final.pt")

        # Final evaluation
        agent.q_net.eval()
        agent.eval(observation, args.episodes, num_episodes=100)

        agent._logger.info("Terminating interaction")
        agent.terminate_connection()
        
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
        experiment_name = "DQN Embed Agent Eval"

        # Load the latest checkpoint
        agent.load("checkpoints/dqn_checkpoint_final.pt")
        agent.q_net.eval()

        agent.eval(observation, args.episodes, 100)
        agent._logger.info("Terminating interaction")
        agent.terminate_connection()
    
    #     with mlflow.start_run(run_name=experiment_name) as run:
    #         # To keep statistics of each episode
    #         wins = 0
    #         detected = 0
    #         max_steps = 0
    #         num_win_steps = []
    #         num_detected_steps = []
    #         num_max_steps_steps = []
    #         num_detected_returns = []
    #         num_win_returns = []
    #         num_max_steps_returns = []

    #         # Log more things in Mlflow
    #         mlflow.set_tag("experiment_name", experiment_name)
    #         # Log notes or additional information
    #         mlflow.set_tag("notes", "This is an evaluation")
    #         mlflow.set_tag("episode_number", args.episodes)
    #         # mlflow.log_param("learning_rate", learning_rate)

    #         for episode in range(1, args.episodes + 1):
    #             agent.logger.info(f"Starting the testing for episode {episode}")
    #             print(f"Starting the testing for episode {episode}")

    #             # Play the game for one episode
    #             observation, num_steps = agent.play_game(observation, 1)

    #             state = observation.state
    #             reward = observation.reward
    #             end = observation.end
    #             info = observation.info

    #             if (
    #                 observation.info
    #                 and observation.info["end_reason"] == AgentStatus.Fail
    #             ):
    #                 detected += 1
    #                 num_detected_steps += [num_steps]
    #                 num_detected_returns += [reward]
    #             elif (
    #                 observation.info
    #                 and observation.info["end_reason"] == AgentStatus.Success
    #             ):
    #                 wins += 1
    #                 num_win_steps += [num_steps]
    #                 num_win_returns += [reward]
    #             elif (
    #                 observation.info
    #                 and observation.info["end_reason"] == AgentStatus.TimeoutReached
    #             ):
    #                 max_steps += 1
    #                 num_max_steps_steps += [num_steps]
    #                 num_max_steps_returns += [reward]

    #             # Reset the game
    #             observation = agent.request_game_reset()

    #             eval_win_rate = (wins / episode) * 100
    #             eval_detection_rate = (detected / episode) * 100
    #             eval_average_returns = np.mean(
    #                 num_detected_returns + num_win_returns + num_max_steps_returns
    #             )
    #             eval_std_returns = np.std(
    #                 num_detected_returns + num_win_returns + num_max_steps_returns
    #             )
    #             eval_average_episode_steps = np.mean(
    #                 num_win_steps + num_detected_steps + num_max_steps_steps
    #             )
    #             eval_std_episode_steps = np.std(
    #                 num_win_steps + num_detected_steps + num_max_steps_steps
    #             )
    #             eval_average_win_steps = np.mean(num_win_steps)
    #             eval_std_win_steps = np.std(num_win_steps)
    #             eval_average_detected_steps = np.mean(num_detected_steps)
    #             eval_std_detected_steps = np.std(num_detected_steps)
    #             eval_average_max_steps_steps = np.mean(num_max_steps_steps)
    #             eval_std_max_steps_steps = np.std(num_max_steps_steps)

    #             # Log and report every X episodes
    #             if episode % args.test_each == 0 and episode != 0:
    #                 text = f"""Tested after {episode} episodes.
    #                     Wins={wins},
    #                     Detections={detected},
    #                     winrate={eval_win_rate:.3f}%,
    #                     detection_rate={eval_detection_rate:.3f}%,
    #                     average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
    #                     average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
    #                     average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
    #                     average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
    #                     average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
    #                     """
    #                 agent.logger.info(text)
    #                 # Store in mlflow
    #                 wandb.log("eval_avg_win_rate", eval_win_rate, step=episode)
    #                 wandb.log(
    #                     "eval_avg_detection_rate", eval_detection_rate, step=episode
    #                 )
    #                 wandb.log(
    #                     "eval_avg_returns", eval_average_returns, step=episode
    #                 )
    #                 wandb.log(
    #                     "eval_std_returns", eval_std_returns, step=episode
    #                 )
    #                 wandb.log(
    #                     "eval_avg_episode_steps",
    #                     eval_average_episode_steps,
    #                     step=episode,
    #                 )
    #                 wandb.log(
    #                     "eval_std_episode_steps", eval_std_episode_steps, step=episode
    #                 )
    #                 wandb.log(
    #                     "eval_avg_win_steps", eval_average_win_steps, step=episode
    #                 )
    #                 wandb.log(
    #                     "eval_std_win_steps", eval_std_win_steps, step=episode
    #                 )
    #                 wandb.log(
    #                     "eval_avg_detected_steps",
    #                     eval_average_detected_steps,
    #                     step=episode,
    #                 )
    #                 wandb.log(
    #                     "eval_std_detected_steps", eval_std_detected_steps, step=episode
    #                 )
    #                 wandb.log(
    #                     "eval_avg_max_steps_steps",
    #                     eval_average_max_steps_steps,
    #                     step=episode,
    #                 )
    #                 wandb.log(
    #                     "eval_std_max_steps_steps",
    #                     eval_std_max_steps_steps,
    #                     step=episode,
    #                 )

    #         # Log the last final episode when it ends
    #         text = f"""Episode {episode}. Final eval after {episode} episodes, for {args.episodes} steps.
    #             Wins={wins},
    #             Detections={detected},
    #             winrate={eval_win_rate:.3f}%,
    #             detection_rate={eval_detection_rate:.3f}%,
    #             average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
    #             average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
    #             average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
    #             average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
    #             average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
    #             """

    #         agent.logger.info(text)
    #         print(text)
    #         agent._logger.info("Terminating interaction")
    #         agent.terminate_connection()

    #         # Print and log the mlflow experiment ID, run ID, and storage location
    #         experiment_id = run.info.experiment_id
    #         run_id = run.info.run_id
    #         storage_location = (
    #             "locally" if not args.mlflow_url else f"at {args.mlflow_url}"
    #         )
    #         print(f"MLflow Experiment ID: {experiment_id}")
    #         print(f"MLflow Run ID: {run_id}")
    #         print(f"Experiment saved {storage_location}")
    #         agent._logger.info(f"MLflow Experiment ID: {experiment_id}")
    #         agent._logger.info(f"MLflow Run ID: {run_id}")
    #         agent._logger.info(f"Experiment saved {storage_location}")

