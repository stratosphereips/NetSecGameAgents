"""
This agent uses LLm embeddings for the state and the actions
Authors:  Maria Rigaki - maria.rigaki@aic.fel.cvut.cz
"""
import argparse
import sys
from collections import deque, namedtuple
import random

# This is used so the agent can see the environment and game components
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) )))

from env.worlds.network_security_game import NetworkSecurityEnvironment
from env.game_components import Action, ActionType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)
from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance

label_mapper = {
    "FindData":ActionType.FindData,
    "FindServices":ActionType.FindServices,
    "ScanNetwork":ActionType.ScanNetwork,
    "ExploitService":ActionType.ExploitService,
    "ExfiltrateData":ActionType.ExfiltrateData
}

# local_services = ['bash', 'powershell', 'remote desktop service',
# 'windows login', 'can_attack_start_here']
local_services = ['can_attack_start_here']

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

Transition = namedtuple('Transition', ('state_emb', 'action_emb', 'real_emb', 'disc_reward', 'state_vals'))


class ReplayBuffer:
    """
    Store and retrieve the episodic data
    """
    def __init__(self, capacity):
        """Initialize the buffer"""
        self.buffer = deque([], maxlen=capacity)

    def append(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        """
        Sample based on the discounted rewards of each state
        """
        # Add 1 so that the weight sum is always positive
        disc_rewards = [mem.disc_reward+1. for mem in self.buffer]
        batch1 = random.choices(self.buffer, disc_rewards, k=batch_size)
        # batch2 = random.sample(self.buffer, batch_size//2)
        return batch1

    def __len__(self):
        """Return the size of the buffer"""
        return len(self.buffer)


class Policy(nn.Module):
    """
    This is the policy that takes as input the observation embedding
    and outputs an action embedding
    """
    def __init__(self, embedding_size=384, output_size=384):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, 128)
        # self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(128, 64)
        # self.dropout2 = nn.Dropout(p=0.2)
        self.output = nn.Linear(64, output_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, input1):
        """Forward pass"""
        x = self.linear1(input1)
        # x = self.dropout(x)
        x = func.relu(x)
        x = self.linear2(x)
        # x = self.dropout2(x)
        x = func.relu(x)
        return self.output(x)

class Baseline(nn.Module):
    """
    Baseline network that takes a state an calculate the value
    """
    def __init__(self, embedding_size=384):
        super().__init__()

        self.linear1 = nn.Linear(embedding_size, 64)
        # self.dropout = nn.Dropout(p=0.2)
        # self.linear2 = nn.Linear(256, 128)
        # self.dropout2 = nn.Dropout(p=0.2)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        """Forward pass"""
        x = self.linear1(x)
        x = func.relu(x)

        # x = self.linear2(x)
        # x = func.relu(x)

        return self.output_layer(x)

class LLMEmbedAgent:
    """
    An agent for the NetSec Game environemnt that uses LLM embeddings.
    The agent is using the REINFORCE algorithm.
    """
    def __init__(self, game_env, args) -> None:
        """
        Create and initialize the agent and the transformer model.
        """
        self.env = game_env
        self.args = args

        if args.memory_len > 0:
            embedding_size = 2*384
        else:
            embedding_size = 384
        self.num_pca = args.num_pca
        self.transformer_model = SentenceTransformer("all-MiniLM-L12-v2").eval()
        self.all_actions = env.get_all_actions()
        self.num_all_actions = len(self.all_actions)
        all_actions_str = [str(action) for action in self.all_actions]
        all_embeddings = self.transformer_model.encode(all_actions_str)

        if args.num_pca < 384:
            self.pca = PCA(n_components=args.num_pca)
            self.pca.fit(all_embeddings)

        self.policy = Policy(embedding_size=embedding_size, output_size=self.num_pca).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)

        self.baseline = Baseline(embedding_size=embedding_size).to(device)
        self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=args.lr)

        self.gamma = args.gamma
        self.loss_fn = nn.MSELoss(reduction='mean')
        # self.loss_fn = nn.SmoothL1Loss()
        self.summary_writer = SummaryWriter()
        self.eval_episodes = args.eval_episodes
        self.memory_len = args.memory_len
        # Parameter that defines the value of the intrinsic reward
        self.beta = 1.0
        self.replay_buffer = ReplayBuffer(args.buffer_size)

    def _create_status_from_state(self, state):
        """
        Create a status prompt using the current state.
        """
        contr_hosts = [host.ip for host in state.controlled_hosts]
        known_hosts = [host.ip for host in state.known_hosts]
        known_nets = [str(net) for net in list(state.known_networks)]

        prompt = "You are a pentester and your goal is to exfiltrate data to host 213.47.23.195.\n"
        prompt += f"Controlled hosts are {' and '.join(contr_hosts)}\n"
        prompt += f"Known networks are {' and '.join(known_nets)}\n"
        prompt += f"Known hosts are {' and '.join(known_hosts)}\n"

        if len(state.known_services.keys()) == 0:
            prompt += "Known services are none\n"
        for ip_service in state.known_services:
            services = []
            if len(list(state.known_services[ip_service])) > 0:
                for serv in state.known_services[ip_service]:
                    if serv.name not in local_services:
                        services.append(serv.name)
                if len(services) > 0:
                    serv_str = ""
                    for serv in services:
                        serv_str += serv + " and "
                    prompt += f"Known services for host {ip_service} are {serv_str}\n"
                else:
                    prompt += "Known services are none\n"

        if len(state.known_data.keys()) == 0:
            prompt += "Known data are none\n"
        for ip_data in state.known_data:
            if len(state.known_data[ip_data]) > 0:
                host_data = ""
                for known_data in list(state.known_data[ip_data]):
                    host_data += f"({known_data.owner}, {known_data.id}) and "
                prompt += f"Known data for host {ip_data} are {host_data}\n"

        return prompt
    
    def _create_status_from_state2(self, state):
        """
        Create a status prompt using the current state.
        """
        contr_hosts = [host.ip for host in state.controlled_hosts]
        known_hosts = [host.ip for host in state.known_hosts]
        known_nets = [str(net) for net in list(state.known_networks)]

        prompt = "You are a pentester and your goal is to exfiltrate data to host 213.47.23.195.\n"
        for host in contr_hosts:
            prompt += f"Controlled host: {host}\n"
        
        for net in known_nets:
            prompt += f"Known network: {net}\n"

        for host in known_hosts:
            prompt += f"Known host {host}\n"

        if len(state.known_services.keys()) == 0:
            prompt += "Known services are none\n"
        for ip_service in state.known_services:
            services = []
            if len(list(state.known_services[ip_service])) > 0:
                for serv in state.known_services[ip_service]:
                    if serv.name not in local_services:
                        services.append(serv.name)
                if len(services) > 0:
                    for serv in services:
                        # serv_str += serv + " and "
                        prompt += f"Known service for host {ip_service}: {serv}\n"
                else:
                    prompt += "Known services are none\n"

        if len(state.known_data.keys()) == 0:
            prompt += "Known data are none\n"
        for ip_data in state.known_data:
            if len(state.known_data[ip_data]) > 0:
                host_data = ""
                for known_data in list(state.known_data[ip_data]):
                    host_data = f"({known_data.owner}, {known_data.id})"
                    prompt += f"Known data for host {ip_data} are {host_data}\n"

        return prompt

    def _create_memory_prompt(self, memory_list):
        """
        Create a string that contains the past actions and their parameters.
        """
        prompt = "Memories:\n"
        if len(memory_list) > 0:
            for memory in memory_list:
                prompt += f'You have taken action {{"action":"{str(memory.type)}", "parameters":"{str(memory.parameters)}"}} in the past.\n'
        else:
            prompt += "No memories yet."
        return prompt

    def _convert_embedding_to_action_pca(self, new_action_embedding, valid_actions, train=True):
        """
        Take an embedded action in the projected space and find the closest
        from the valid actions.
        """
        all_actions_str = [str(action) for action in valid_actions]
        valid_embeddings = self.transformer_model.encode(all_actions_str)
        valid_pca = self.pca.transform(valid_embeddings)
        # dist = cosine_distances(valid_pca, new_action_embedding).flatten()
        # dist = euclidean_distances(valid_pca, new_action_embedding).flatten()
        dist = [distance.chebyshev(vector.flatten(), new_action_embedding.flatten()) for vector in valid_pca]
        dist = np.array(dist).flatten()
        eps = np.finfo(np.float32).eps.item()
        if train:
            # action_id = random.choices(population=range(len(valid_pca)), weights=1.0/dist, k=1)[0]
            action_id = random.choices(population=range(len(valid_pca)), weights=1.0/(eps+dist), k=1)[0]
        else:
            # TODO: Select one of the top-3 actions
            action_id = np.argmin(dist, axis=0)

        return valid_actions[action_id], valid_pca[action_id]

    def _convert_embedding_to_action(self, new_action_embedding, valid_actions, train=True):
        """
        Take an embedding, and the valid actions for the state
        and find the closest embedding using cosine similarity
        Return an Action object and the closest neighbor
        """
        all_actions_str = [str(action) for action in valid_actions]
        valid_embeddings = self.transformer_model.encode(all_actions_str)
        # dists = cosine_distances(valid_embeddings, new_action_embedding).flatten()
        dists = euclidean_distances(valid_embeddings, new_action_embedding).flatten()
        eps = np.finfo(np.float32).eps.item()
        if train:
            action_id = random.choices(population=range(len(valid_actions)), weights=1.0/(eps+dists), k=1)[0]
        else:
            action_id = np.argmin(dists, axis=0)

        return valid_actions[action_id], valid_embeddings[action_id]

    def _generate_valid_actions(self, state):
        """
        Generate a list of valid actions from the current state.
        """
        valid_actions = set()
        #Network Scans
        for network in state.known_networks:
            valid_actions.add(Action(ActionType.ScanNetwork, params={"target_network": network}))
        # Service Scans
        for host in state.known_hosts:
            valid_actions.add(Action(ActionType.FindServices, params={"target_host": host}))
        # Service Exploits
        for host, service_list in state.known_services.items():
            for service in service_list:
                valid_actions.add(Action(ActionType.ExploitService,
                                         params={"target_host": host , "target_service": service}))
        # Data Scans
        for host in state.controlled_hosts:
            valid_actions.add(Action(ActionType.FindData, params={"target_host": host}))

        # Data Exfiltration
        for src_host, data_list in state.known_data.items():
            for data in data_list:
                for trg_host in state.controlled_hosts:
                    if trg_host != src_host:
                        valid_actions.add(Action(
                            ActionType.ExfiltrateData, params={"target_host": trg_host,
                                                               "source_host": src_host, 
                                                               "data": data}))
        return list(valid_actions)

    def _filter_valid_actions(self, valid_actions, memory):
        """
        Remove taken actions from the list of valid actions.
        """
        if len(memory) == 0:
            return valid_actions
        filtered_actions = []
        common_actions = list(set(valid_actions).intersection(memory))
        for action in valid_actions:
            if action not in common_actions:
                filtered_actions.append(action)
        return filtered_actions

    def _weight_histograms_linear(self, step, weights, layer_name):
        """
        Log the histograms of the weight of a specific layer to tensorboard
        """
        flattened_weights = weights.flatten()
        tag = f"policy_layer_{layer_name}"
        self.summary_writer.add_histogram(tag,
                                          flattened_weights,
                                          global_step=step,
                                          bins='tensorflow')

    def _weight_histograms(self, step):
        """
        Go over each layer and if it is a linear layer send it to the
        logger function.
        """
        # Iterate over all model layers
        for layer_name in self.policy._modules.keys():
            layer = self.policy._modules[layer_name]
            # Compute weight histograms for appropriate layer
            if isinstance(layer, nn.Linear):
                weights = layer.weight
                self._weight_histograms_linear(step, weights, layer_name)

    def _get_discounted_rewards(self, rewards):
        """
        Calculate the return G
        """
        returns = deque()

        for time_step in range(len(rewards))[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(self.gamma*disc_return_t + rewards[time_step])

        eps = np.finfo(np.float32).eps.item()
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        return returns

    def _training_step(self, returns, out_embeddings, real_embeddings, episode):
        """
        Backpropagation step for the policy network.
        """
        # Calculate the discounted rewards
        policy_losses = []

        for out_emb, real_emb, disc_ret in zip(out_embeddings, real_embeddings, returns):
            # rmse_loss = torch.sqrt(self.loss_fn(out_emb, real_emb))
            mse_loss = self.loss_fn(out_emb, real_emb)
            policy_losses.append((-mse_loss * disc_ret).reshape(1))

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_losses).sum()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)

        self.optimizer.step()

        for tag, param in self.policy.named_parameters():
            self.summary_writer.add_histogram(f"grad_{tag}", param.grad.data.cpu().numpy(), episode)

        return policy_loss.item()

    def _training_step_baseline(self, state_vals, returns, episode):
        """
        Backpropagation step for the baseline network.
        """
        # Calculate MSE loss
        value_loss = func.mse_loss(state_vals.squeeze(), returns)

        self.baseline_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)

        # torch.nn.utils.clip_grad_value_(self.policy.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        torch.nn.utils.clip_grad_value_(self.baseline.parameters(), 100)
        self.baseline_optimizer.step()

        for tag, param in self.baseline.named_parameters():
            self.summary_writer.add_histogram(f"baseline_grad_{tag}", param.grad.data.cpu().numpy(), episode)

        return value_loss.item()

    def _optimize_models(self, episode):
        """
        Run the training steps for a number of epochs using the replay buffer.
        """
        value_loss = 0
        policy_loss = 0
        if len(self.replay_buffer) < self.args.batch_size:
            return
        for _ in range(self.args.num_epochs):
            transitions = self.replay_buffer.sample(self.args.batch_size)

            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Train the baseline first
            values_batch = torch.cat(batch.state_vals)
            reward_batch = torch.cat(batch.disc_reward)
            action_batch = torch.cat(batch.action_emb)
            real_batch = torch.cat(batch.real_emb)

            value_loss = value_loss + self._training_step_baseline(values_batch, reward_batch, episode)

            # Calculate deltas and train the policy network
            deltas = [gt - val for gt, val in zip(reward_batch, values_batch)]
            deltas = torch.tensor(deltas).to(device)
            policy_loss = policy_loss + self._training_step(deltas, action_batch, real_batch, episode)

        self.summary_writer.add_scalar("losses/value_loss", value_loss/self.args.num_epochs, episode)
        self.summary_writer.add_scalar("losses/policy_loss", policy_loss/self.args.num_epochs, episode)

    def train(self, num_episodes):
        """
        Main training loop that runs for a number of episodes.
        """
        scores = []
        scores_int = []
        self.policy.train()
        self.baseline.train()

        # Keep track of the wins during training
        wins = 0
        for episode in range(1, num_episodes+1):
            input_embeddings = []
            out_embeddings = []
            real_embeddings = []
            rewards = []
            intr_rewards = []
            memories = []
            filtered_actions = []

            state_vals = []
            observation = self.env.reset()
            valid_actions = self._generate_valid_actions(observation.state)
            num_valid = len(valid_actions)

            # Visualize the weights in tensorboard
            self._weight_histograms(episode)
            for _ in range(self.args.max_t):
                # Create the status string from the observed state
                status_str = self._create_status_from_state(observation.state)
                # Get the embedding of the string from the transformer
                state_embed = self.transformer_model.encode(status_str)
                state_embed = torch.tensor(state_embed, device=device).unsqueeze(0)

                if self.memory_len > 0:
                    memory_str = self._create_memory_prompt(memories[-self.memory_len:])

                    # Get the embedding of the memory string from the transformer
                    memory_embed = self.transformer_model.encode(memory_str)
                    memory_embed = torch.tensor(memory_embed, device=device).unsqueeze(0)

                    input_emb = torch.concat([state_embed, memory_embed], dim=1)
                    input_embeddings.append(input_emb)

                    # Pass the state embedding to the baseline and get the value
                    state_val = self.baseline.forward(input_emb)

                    # Pass the state embedding to the model and get the action
                    action_emb = self.policy.forward(input_emb)
                else:
                    input_embeddings.append(state_embed)
                    # Pass the state embedding to the model and get the action
                    action_emb = self.policy.forward(state_embed)

                    # Pass the state embedding to the baseline and get the value
                    state_val = self.baseline.forward(state_embed)

                out_embeddings.append(action_emb)
                state_vals.append(state_val)

                # Convert the action embedding to a valid action and its embedding
                # remaining_actions = self._filter_valid_actions(valid_actions, filtered_actions)
                remaining_actions = valid_actions
                # print(remaining_actions, memories)
                if self.num_pca < 384:
                    action, real_emb = self._convert_embedding_to_action_pca(action_emb.cpu().detach().numpy(), remaining_actions, True)
                else:
                    action, real_emb = self._convert_embedding_to_action(action_emb.cpu().detach().numpy(), remaining_actions, True)
                real_embeddings.append(torch.tensor(real_emb, device=device).unsqueeze(0))

                # Take the new action and get the observation from the environment
                observation = self.env.step(action)
                rewards.append(observation.reward)

                # Add an intrinsic reward based on the number of valid actions
                valid_actions = self._generate_valid_actions(observation.state)
                if len(valid_actions) > num_valid:
                    num_valid = len(valid_actions)
                    intr_rewards.append(4.0)
                    filtered_actions.append(action)
                else:
                    intr_rewards.append(0.0)
                    if action in memories:
                        filtered_actions.append(action)

                memories.append(action)

                if observation.done:
                    # If done and the latest reward from the env is positive, we have reached the goal
                    if observation.reward > 0:
                        wins += 1
                    break

            scores.append(sum(rewards))
            scores_int.append(sum(intr_rewards))

            self.summary_writer.add_scalar("actions/valid_actions", len(valid_actions), episode)
            self.summary_writer.add_scalar("reward/mean_ext", np.mean(scores), episode)
            self.summary_writer.add_scalar("reward/mean_int", np.mean(scores_int), episode)
            self.summary_writer.add_scalar("reward/moving_average_ext", np.mean(scores[-self.args.eval_every:]), episode)
            self.summary_writer.add_scalar("wins/num_wins", wins, episode)
            self.summary_writer.add_scalar("wins/win_rate", 100*(wins/episode), episode)

            # Calulated discounted rewards for the current episode
            returns = self._get_discounted_rewards(rewards).to(device)
            intr_returns = self._get_discounted_rewards(intr_rewards).to(device)
            total_returns = returns+self.beta*intr_returns

            # Populate the replay buffer
            for i in range(self.args.max_t):
                try:
                    self.replay_buffer.append(input_embeddings[i], out_embeddings[i], real_embeddings[i], total_returns[i].reshape(1), state_vals[i])
                except:
                    break

            if episode > 0 and episode % self.args.train_every == 0:
                self._optimize_models(episode)

            if episode > 0 and episode % self.args.eval_every == 0:
                eval_rewards, eval_wins = self.evaluate(args.eval_episodes)
                self.summary_writer.add_scalar('test/eval_rewards', np.mean(eval_rewards), episode)
                self.summary_writer.add_scalar('test/wins', eval_wins, episode)

    def evaluate(self, num_eval_episodes):
        """
        Evaluation function.
        """
        self.policy.eval()
        eval_returns = []
        wins = 0
        for _ in range(num_eval_episodes):
            observation = env.reset()
            done = False
            ret = 0
            memories = []
            filtered_actions = []

            valid_actions = self._generate_valid_actions(observation.state)
            num_valid = len(valid_actions)
            while not done:
                # Create the status string from the observed state
                status_str = self._create_status_from_state(observation.state)
                # Get the embedding of the string from the transformer
                state_embed = self.transformer_model.encode(status_str)
                state_embed_t = torch.tensor(state_embed, device=device).unsqueeze(0)

                if self.memory_len > 0:
                    memory_str = self._create_memory_prompt(memories[-self.memory_len:])

                    # Get the embedding of the memory string from the transformer
                    memory_embed = self.transformer_model.encode(memory_str)
                    memory_embed_t = torch.tensor(memory_embed, device=device).unsqueeze(0)

                    input_emb = torch.concat([state_embed_t, memory_embed_t], dim=1)

                    # Pass the state embedding to the model and get the action
                    action_emb = self.policy.forward(input_emb)
                else:
                    action_emb = self.policy.forward(state_embed_t)

                # Filter the actions already taken successfully
                # remaining_actions = self._filter_valid_actions(valid_actions, filtered_actions)
                remaining_actions = valid_actions

                # Convert the action embedding to a valid action and its embedding
                if self.num_pca < 384:
                    action, _ = self._convert_embedding_to_action_pca(action_emb.cpu().detach().numpy(), remaining_actions, False)
                else:
                    action, _ = self._convert_embedding_to_action(action_emb.cpu().detach().numpy(), remaining_actions, False)

                # Take the new action and get the observation from the policy
                observation = self.env.step(action)
                ret += observation.reward
                done = observation.done

                # Generate the list of all the valid actions in the observed state
                valid_actions = self._generate_valid_actions(observation.state)
                if len(valid_actions) > num_valid:
                    num_valid = len(valid_actions)
                    # If the action is not successful do not retake it
                    filtered_actions.append(action)
                else:
                    if action in memories:
                        filtered_actions.append(action)

                # Finally, add the action to the memory list
                memories.append(action)

            if observation.reward > 0:
                wins += 1

            eval_returns.append(ret)

        self.policy.train()
        return eval_returns, wins

    def save_model(self, file_name):
        """
        Save the pytorch policy model.
        """
        torch.save(self.policy.state_dict(), file_name)

    def load_model(self, file_name):
        """
        Load the model
        """
        self.policy = Policy(embedding_size=2*384, output_size=10).to(device)
        self.policy.load_state_dict(torch.load(file_name))
        self.policy.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Task config file
    parser.add_argument("--task_config_file",
                        help="Reads the task definition from a configuration file",
                        default=path.join(path.dirname(__file__), 'netsecenv-task.yaml'),
                        action='store',
                        required=False)

    # Model arguments
    parser.add_argument("--gamma", help="Sets gamma for discounting", default=0.9, type=float)
    parser.add_argument("--lr", help="Learning rate of the NN", type=float, default=1e-3)
    parser.add_argument("--memory_len", type=int, default=0, help="Number of memories to keep. Zero means no memory")
    parser.add_argument("--model_path", type=str, default="saved_models/policy.pt", help="Path for saving the policy model.")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the networks")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to sample from memory")
    parser.add_argument("--num_episodes", help="Sets number of training episodes", default=1000, type=int)
    parser.add_argument("--max_t", type=int, default=128, help="Max episode length")
    parser.add_argument("--train_every", type=int, default=4, help="Train every this ammount of episodes.")
    parser.add_argument("--eval_every", help="During training, evaluate every this amount of episodes.", default=128, type=int)
    parser.add_argument("--eval_episodes", help="Sets evaluation length", default=100, type=int)
    parser.add_argument("--num_pca", type=int, default=10, help="Number of PCA components. Use 384 to disable PCA")
    parser.add_argument("--buffer_size", type=int, default=128, help="Replay buffer size")
    # parser.add_argument("--top_k", type=int, default=5, help="The number of valid actions to consider for similarity")

    args = parser.parse_args()

    # Create the environment
    env = NetworkSecurityEnvironment(args.task_config_file)

    # Initializr the agent
    agent = LLMEmbedAgent(env, args)

    # Train the agent using reinforce
    agent.train(args.num_episodes)

    # Evaluate the agent
    final_returns, wins = agent.evaluate(args.eval_episodes)
    print(f"Evaluation finished - (mean of {len(final_returns)} runs): {np.mean(final_returns)}+-{np.std(final_returns)}")
    print(f"Total number of wins during evaluation: {wins}")
    agent.save_model(args.model_path)

    # agent.load_model('saved_models/winning.pt')
    # final_returns, wins = agent.evaluate(args.eval_episodes)
    # print(f"Evaluation finished - (mean of {len(final_returns)} runs): {np.mean(final_returns)}+-{np.std(final_returns)}")
    # print(f"Total number of wins during evaluation: {wins}")
    
