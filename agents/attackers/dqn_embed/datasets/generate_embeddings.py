# Run the random agent to generate embeddings for all states in the environment
import logging
import argparse
import numpy as np
from os import path, makedirs
from random import choice
from AIDojoCoordinator.game_components import Action, Observation
from NetSecGameAgents.agents.base_agent import BaseAgent
from NetSecGameAgents.agents.agent_utils import generate_valid_actions, filter_log_files_from_state, state_as_ordered_string, convert_ips_to_concepts

from sentence_transformers import SentenceTransformer, models
import torch
import json

from transformers import AutoTokenizer, AutoModel
import torch

# class TextEncoderCodeBert:
#     def __init__(self, model_name="microsoft/codebert-base"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)

#         # self.embedding_dim = self.model.get_sentence_embedding_dimension()
#         self.model.eval()  # Set the model to evaluation mode

#     def encode(self, input_data):
#         if isinstance(input_data, dict):
#             input_str = json.dumps(input_data, sort_keys=True)
#         elif isinstance(input_data, str):
#             input_str = input_data
#         else:
#             raise ValueError("Unsupported input type")
#         with torch.no_grad():
#             code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(input_str) + [self.tokenizer.eos_token]
#             print(code_tokens)
#             tokens_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
#             print("Token IDs:", tokens_ids)
#             embedding = self.model(torch.tensor(tokens_ids)[None,:])[0].squeeze(0).mean(dim=0).float().cpu()
#             print("Embedding shape:", embedding.shape)
#         return embedding


class TextEncoder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model.eval()  # Set the model to evaluation mode

    def encode(self, input_data):
        if isinstance(input_data, dict):
            input_data = json.dumps(input_data, sort_keys=True)
        elif isinstance(input_data, str):
            input_data = input_data
        else:
            raise ValueError("Unsupported input type")
        with torch.no_grad():
            embedding = self.model.encode(input_data)
        return torch.tensor(embedding).float()


class RandomAttackerAgent(BaseAgent):

    def __init__(self, host, port,role, seed) -> None:
        super().__init__(host, port, role)
        self.state_encoder = TextEncoder()
        self.action_encoder = TextEncoder()

        self.state_embeddings = dict()
        self.action_embeddings = dict()

    def extract_sentences_from_state(self, json_str: str) -> list:
        state_dict = json.loads(json_str)
        sentences = []
        for key, value in state_dict.items():
            sentences.append(f"{key}: {value}")

        return sentences
    
    def generate_state_embedding(self, observation:Observation):
        obs = filter_log_files_from_state(observation)

        state_str = obs.state.as_json()
        state_key = state_str
        
        if state_key not in self.state_embeddings.keys():

            embedding = self.state_encoder.encode(state_str).detach().float().cpu()
            self.state_embeddings[state_key] = embedding

        
    def generate_action_embedding(self, action:Action):
        action_str = str(action)
        if action_str not in self.action_embeddings.keys():
            embedding = self.state_encoder.encode(action_str).detach().float().cpu()
            self.action_embeddings[action_str] = embedding
        # return embedding


    def save_embeddings(self, file_name:str, my_dict: dict):
        state_names = list(my_dict.keys())
        embeddings_matrix = np.vstack(list(my_dict.values()))

        np.savez_compressed(
           file_name,
           embeddings=embeddings_matrix,
           states=state_names
        )

    def play_game(self, observation, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        returns = []
        
        num_steps = 0
        for episode in range(num_episodes):
            self._logger.info(f"Playing episode {episode}")
            episodic_returns = []
            while observation and not observation.end:

                self.generate_state_embedding(observation)
                num_steps += 1
                self._logger.debug(f'Observation received:{observation}')
                # Store returns in the episode
                episodic_returns.append(observation.reward)
                # Select the action randomly
                action = self.select_action(observation)
                # self.generate_action_embedding(action)
                observation = self.make_step(action)
                # To return
                last_observation = observation
            self._logger.debug(f'Observation received:{observation}')
            returns.append(np.sum(episodic_returns))
            self._logger.info(f"Episode {episode} ended with return{np.sum(episodic_returns)}. Mean returns={np.mean(returns)}±{np.std(returns)}")
            # Reset the episode
            observation = self.request_game_reset()
        self._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        self._logger.info(f"Length of state embeddings: {len(self.state_embeddings)} and action embeddings: {len(self.action_embeddings)}")
        self.save_embeddings("state_embeddings_random_agent_json_4.npz", self.state_embeddings)
        # self.save_embeddings("action_embeddings_random_agent_scen_3.npz", self.action_embeddings)
        # This will be the last observation played before the reset
        return (last_observation, num_steps)
    
    def select_action(self, observation:Observation)->Action:
        valid_actions = generate_valid_actions(observation.state)
        action = choice(valid_actions)
        return action

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to play or evaluate", default=100, type=int) 
    parser.add_argument("--test_each", help="Evaluate performance during testing every this number of episodes.", default=10, type=int)
    parser.add_argument("--logdir", help="Folder to store logs", default=path.join(path.dirname(path.abspath(__file__)), "logs"))
    parser.add_argument("--mlflow_url", help="URL for mlflow tracking server. If not provided, mlflow will store locally.", default=None)
    args = parser.parse_args()

    if not path.exists(args.logdir):
        makedirs(args.logdir)
    logging.basicConfig(filename=path.join(args.logdir, "random_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Create agent
    agent = RandomAttackerAgent(args.host, args.port,"Attacker", seed=42)

    # Play the normal game
    observation = agent.register()
    agent.play_game(observation, args.episodes)
    agent._logger.info("Terminating interaction")
    agent.terminate_connection()
    agent._logger.info("Interaction terminated")