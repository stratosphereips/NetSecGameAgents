#Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# This agents just randomnly picks actions. No learning
import logging
import argparse
import numpy as np
import mlflow
from os import path, makedirs
import random
from netsecgame import Action, Observation, BaseAgent, generate_valid_actions, AgentRole
from netsecgame.game_components import AgentStatus

class RandomAttackerAgent(BaseAgent):
    """
    An attacker agent that selects actions randomly without learning.
    Inherits from BaseAgent.
    """

    def __init__(self, host, port, role, seed) -> None:
        """
        Initialize the RandomAttackerAgent.
        
        Args:
            host (str): Host address to connect to.
            port (int): Port number to connect to.
            role (AgentRole): The role of the agent (e.g., AgentRole.Attacker).
            seed (int): Seed for random number generation for the agent's decisions.
        """
        super().__init__(host, port, role)
        self.rng = random.Random(seed)

    def select_action(self, observation:Observation)->Action:
        """
        Selects a random action from the set of valid actions in the current state.
        
        Args:
            observation (Observation): The current observation including the game state.
            
        Returns:
            Action: The randomly selected action.
        """
        valid_actions = generate_valid_actions(observation.state)
        # randomly choose with the seeded rng
        action = self.rng.choice(valid_actions)
        return action
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to play", default=100, type=int)
    parser.add_argument("--seed", help="Sets random seed for agent's decisions", default=42, type=int) 
    parser.add_argument("--logdir", help="Folder to store logs", default=path.join(path.dirname(path.abspath(__file__)), "logs"))
    parser.add_argument("--mlflow_url", help="URL for mlflow tracking server. If not provided, mlflow will store locally.", default=None)
    args = parser.parse_args()

    if not path.exists(args.logdir):
        makedirs(args.logdir)
    logging.basicConfig(filename=path.join(args.logdir, "random_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Create agent
    agent = RandomAttackerAgent(args.host, args.port, AgentRole.Attacker, seed=args.seed)

    # Mlflow experiment name        
    experiment_name = "Random Attacker Agent"
    if args.mlflow_url:
        mlflow.set_tracking_uri(args.mlflow_url)
    mlflow.set_experiment(experiment_name)
    
    # Register in the game
    observation = agent.register()
    
    with mlflow.start_run(run_name=experiment_name) as run:
        # To keep statistics of each episode
        wins = 0
        detected = 0
        max_steps = 0
        num_win_steps = []
        num_detected_steps = []
        num_max_steps_steps = []
        num_detected_returns = []
        num_win_returns = []
        num_max_steps_returns = []

        # Log more things in Mlflow
        mlflow.set_tag("experiment_name", experiment_name)
        mlflow.set_tag("episode_number", args.episodes)

        for episode in range(1, args.episodes + 1):
            agent.logger.info(f'Starting episode {episode}')
            print(f'Starting episode {episode}')

            # Play the game for one episode
            episodic_returns = []
            num_steps = 0
            
            while observation and not observation.end:
                num_steps += 1
                agent.logger.debug(f'Observation received:{observation}')
                # Store returns in the episode
                episodic_returns.append(observation.reward)
                # Select the action randomly
                action = agent.select_action(observation)
                observation = agent.make_step(action)
            
            agent.logger.debug(f'Observation received:{observation}')
            current_return = np.sum(episodic_returns)
            
            agent.logger.info(f"Episode {episode} ended with return {current_return}.")
            
            reward = current_return
            
            if observation.info and observation.info.get('end_reason') == AgentStatus.Fail:
                detected +=1
                num_detected_steps.append(num_steps)
                num_detected_returns.append(reward)
            elif observation.info and observation.info.get('end_reason') == AgentStatus.Success:
                wins += 1
                num_win_steps.append(num_steps)
                num_win_returns.append(reward)
            elif observation.info and observation.info.get('end_reason') == AgentStatus.TimeoutReached:
                max_steps += 1
                num_max_steps_steps.append(num_steps)
                num_max_steps_returns.append(reward)

            # Reset the game - ONLY ONCE
            if episode < args.episodes:
                 observation = agent.request_game_reset()

            # Calculate stats for logging
            eval_win_rate = (wins/episode) * 100
            eval_detection_rate = (detected/episode) * 100
            
            all_returns = num_detected_returns + num_win_returns + num_max_steps_returns
            eval_average_returns = np.mean(all_returns) if all_returns else 0
            eval_std_returns = np.std(all_returns) if all_returns else 0
            
            all_steps = num_win_steps + num_detected_steps + num_max_steps_steps
            eval_average_episode_steps = np.mean(all_steps) if all_steps else 0
            eval_std_episode_steps = np.std(all_steps) if all_steps else 0

            # Store in mlflow
            mlflow.log_metric("win_rate", eval_win_rate, step=episode)
            mlflow.log_metric("detection_rate", eval_detection_rate, step=episode)
            mlflow.log_metric("avg_returns", eval_average_returns, step=episode)
            mlflow.log_metric("std_returns", eval_std_returns, step=episode)
            mlflow.log_metric("avg_episode_steps", eval_average_episode_steps, step=episode)
        
        # Log the last final episode when it ends
        text = f'''Final results for {args.episodes} episodes:
            Wins={wins},
            Detections={detected},
            MaxSteps={max_steps},
            winrate={eval_win_rate:.3f}%,
            detection_rate={eval_detection_rate:.3f}%,
            average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
            average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f}
            '''

        agent.logger.info(text)
        print(text)
        agent._logger.info("Terminating interaction")
        agent.terminate_connection()

        # Print and log the mlflow experiment ID, run ID, and storage location
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id
        storage_location = "locally" if not args.mlflow_url else f"at {args.mlflow_url}"
        print(f"MLflow Experiment ID: {experiment_id}")
        print(f"MLflow Run ID: {run_id}")
        print(f"Experiment saved {storage_location}")
        agent._logger.info(f"MLflow Experiment ID: {experiment_id}")
        agent._logger.info(f"MLflow Run ID: {run_id}")
        agent._logger.info(f"Experiment saved {storage_location}")

if __name__ == '__main__':
    main()