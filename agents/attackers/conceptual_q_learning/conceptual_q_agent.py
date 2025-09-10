# Authors:    Sebastian Garcia. sebastian.garcia@agents.fel.cvut.cz
# This is the conceptual Q-learning attacking agent.
# It uses concepts instead of IP addresses to learn the Q-table.

from collections import namedtuple
import sys
import numpy as np
import random
import pickle
import argparse
import logging
import subprocess
import time
import wandb

from os import path, makedirs
# with the path fixed, we can import now
from AIDojoCoordinator.game_components import Action, Observation, GameState, AgentStatus, ActionType
from NetSecGameAgents.agents.base_agent import BaseAgent
from NetSecGameAgents.agents.agent_utils import state_as_ordered_string, convert_ips_to_concepts, convert_concepts_to_actions, generate_valid_actions_concepts
from NetSecGameAgents.utils.concept_mapping_logger import ConceptMappingLogger

class QAgent(BaseAgent):

    def __init__(self, host, port, role="Attacker", alpha=0.1, gamma=0.6, epsilon_start=0.9, epsilon_end=0.1, epsilon_max_episodes=5000, apm_limit:int=None) -> None:
        super().__init__(host, port, role)
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self._str_to_id = {}
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_max_episodes = epsilon_max_episodes
        self.current_epsilon = epsilon_start
        self._apm_limit = apm_limit
        if self._apm_limit:
            self.inter_action_interval = 60/apm_limit
        else:
            self.inter_action_interval = 0
        # Store the concepts that got acted on
        self.actions_history = set()
        # store the last state seen, so we know if there was a change or not
        self.previous_state = None
        # Enhanced logging
        self.concept_logger = None

    def store_q_table(self, strpath, filename):
        """ Store the q table on disk """
        # path.join(path.dirname(path.abspath(__file__)), "logs")
        if not path.exists(strpath):
            makedirs(strpath)
        with open(strpath+filename, "wb") as f:
            data = {"q_table":self.q_values, "state_mapping": self._str_to_id}
            pickle.dump(data, f)

    def load_q_table(self,filename):
        """ Load the q table from disk """
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.q_values = data["q_table"]
                self._str_to_id = data["state_mapping"]
            self._logger.info(f'Successfully loading file {filename}')
        except Exception as e:
            self._logger.info(f'Error loading file {filename}. {e}')
            sys.exit(-1)

    def get_state_id(self, state:GameState) -> int:
        """ For each state, get a unique id number """
        # Here the state has to be ordered, so different orders are not taken as two different states.
        state_str = state_as_ordered_string(state)
        if state_str not in self._str_to_id:
            self._str_to_id[state_str] = len(self._str_to_id) 
        return self._str_to_id[state_str]
    
    def max_action_q(self, concept_observation:Observation) -> Action:
        """ Get the action that maximices the q_value for a given observation """
        state = concept_observation.observation.state
        actions = generate_valid_actions_concepts(state, self.actions_history)
        state_id = self.get_state_id(state)
        tmp = dict(((state_id, a), self.q_values.get((state_id, a), 0)) for a in actions)
        if not tmp:
            # tmp is empty, meaning there are no actions to take!
            return None
        return tmp[max(tmp,key=tmp.get)] #return maximum Q_value for a given state (out of available actions)
   
    def select_action(self, observation:Observation, testing=False) -> tuple:
        """ Select the action according to the algorithm """
        state = observation.state
        actions = generate_valid_actions_concepts(state, self.actions_history)
        state_id = self.get_state_id(state)
        
        # E-greedy play. If the random number is less than the e, then choose random to explore.
        # But do not do it if we are testing a model. In testing is always exploit so it is deterministic. 
        # Epsilon 0 means only exploit, which is very good if the env does not change.
        if random.uniform(0, 1) <= self.current_epsilon and not testing:
            # We are training
            # Random choose an ation from the list of actions?
            action = random.choice(list(actions))
            if (state_id, action) not in self.q_values:
                self.q_values[state_id, action] = 0
            return action, state_id
        else: 
            # Here we can be during training outside the e-greede, or during testing
            # Select the action with highest q_value, or random pick to break the ties
            # The default initial q-value for a (state, action) pair is 0.
            initial_q_value = 0
            tmp = dict(((state_id, action), self.q_values.get((state_id, action), initial_q_value)) for action in actions)
            ((state_id, action), value) = max(tmp.items(), key=lambda x: (x[1], random.random()))
            try:
                self.q_values[state_id, action]
            except KeyError:
                self.q_values[state_id, action] = 0
            return action, state_id

    def enable_enhanced_logging(self, verbose=True):
        """Enable enhanced concept mapping logging"""
        self.concept_logger = ConceptMappingLogger(self._logger, verbose)

    def recompute_reward(self, observation: Observation) -> Observation:
        """
        Redefine how this agent recomputes its inner reward
        """
        state = observation.state
        end = observation.end
        info = observation.info
        reward = observation.reward # Just to pass it over in the first observation

        # The first observation from the env does not have an end reson yet
        try:
            if info and info['end_reason'] == AgentStatus.Fail:
                reward = -1000
            elif info and info['end_reason'] == AgentStatus.Success:
                reward = 1000
            elif info and info['end_reason'] == AgentStatus.TimeoutReached:
                reward = -100
            elif state == self.previous_state: # This is not good and the agent should learn to avoid these actions
                reward = -100
            else:
                reward = -1
            self.previous_state = state
        except KeyError:
            pass

        new_observation = Observation(state, reward, end, info)
        return new_observation

    def update_epsilon_with_decay(self, episode_number)->float:
        """ 
        Decay the epsilon 
        """
        decay_rate = np.max([(self.epsilon_max_episodes - episode_number) / self.epsilon_max_episodes, 0])
        new_eps = (self.epsilon_start - self.epsilon_end ) * decay_rate + self.epsilon_end
        self.logger.debug(f"Updating epsilon - new value:{new_eps}")
        return new_eps
    
    def remember_action(self, concept_action):
        """ 
        Mark the action as done, so it is not repeated
        """
        self.actions_history.add(concept_action)
        
    
    def play_game(self, concept_observation, episode_num, testing=False):
        """
        Only play one episode of the game.

        The main function for the gameplay. Handles the main interaction loop.
        The conversion from IPs to concepts is done here, so the agent can use concepts
        Observation is in concepts
        episode_num is used to update the epsilon value at the end of the episode.
        """
        num_steps = 0

        # Run the whole episode
        while not concept_observation.observation.end:
            # Store steps so far
            num_steps += 1
            if self.concept_logger:
                self.concept_logger.set_episode_step(episode_num, num_steps)
            start_time = time.time()
            # Get next action. If we are not training, selection is different, so pass it as argument
            concept_action, state_id = self.select_action(concept_observation.observation, testing)
            self.logger.info(f"\n\n ==================================== \n\n[+] Concept Action selected:{concept_action}")

            # Convert the action with concepts to the action with IPs
            action = convert_concepts_to_actions(concept_action, concept_observation, self.concept_logger)
            self.logger.info(f"\n[+] Real Action selected:{action}")

            if self.concept_logger:
                self.concept_logger.log_action_history_update(concept_action, self.actions_history)
            self.remember_action(concept_action)

            # Perform the action and observe next observation
            # This observation is in IPs
            observation = self.make_step(action)
            self.logger.info(f"\n[+] State after action:{observation}")

            # Recompute the rewards
            observation = self.recompute_reward(observation)

            # Convert the observation to conceptual observation
            # From now one the observation will be in concepts
            concept_observation = convert_ips_to_concepts(observation, self.logger, self.concept_logger)
           
            #concept_observation = self.recompute_reward(concept_observation)
            self.logger.info(f"\n[+] Reward of last action (after reward engineering): {concept_observation.observation.reward}")

            # Update the Q-table
            if not testing:
                # If we are training update the Q-table. If in testing do not update, so no learning in testing.
                max_action = self.max_action_q(concept_observation)
                if max_action == None:
                    # There are no more actions to take. 
                    self.logger.info(f"\n[+] We run out of actions.")
                    return None, num_steps
                old_q = self.q_values[state_id, concept_action]
                self.q_values[state_id, concept_action] += self.alpha * (concept_observation.observation.reward + max_action) - self.q_values[state_id, concept_action]
                new_q = self.q_values[state_id, concept_action]
                if self.concept_logger:
                    self.concept_logger.log_q_value_update(
                        state_id, concept_action, old_q, new_q,
                        concept_observation.observation.reward, max_action, self.alpha, self.gamma
                    )

            # Check the apm (actions per minute)
            if self._apm_limit:
                elapsed_time = time.time() - start_time
                remaining_time = self.inter_action_interval - elapsed_time
                if remaining_time > 0:
                    # We still have some time in this interval, but we can not
                    # take more actions. So wait until the next interval starts
                    self._logger.debug(f"Waiting for {remaining_time}s before next action.")
                    time.sleep(remaining_time)
                start_time = time.time()

        # update epsilon value
        if not testing:
            self.current_epsilon = self.update_epsilon_with_decay(episode_num)

        # Log episode summary
        if self.concept_logger:
            end_reason = "success" if observation and observation.info and observation.info.get('end_reason') == AgentStatus.Success else \
                        "detected" if observation and observation.info and observation.info.get('end_reason') == AgentStatus.Fail else \
                        "timeout" if observation and observation.info and observation.info.get('end_reason') == AgentStatus.TimeoutReached else "unknown"
            reward = observation.reward if observation else 0
            self.concept_logger.log_episode_summary(
                episode_num, num_steps, reward, len(self.q_values), self.current_epsilon, end_reason
            )

        # This will be the last observation played before the reset
        return observation, num_steps

if __name__ == '__main__':
    parser = argparse.ArgumentParser('You can train the agent, or test it. \n Test is also to use the agent. \n During training and testing the performance is logged.')
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to run.", default=15000, type=int)
    parser.add_argument("--test_each", help="Evaluate the performance every this number of episodes. During training and testing.", default=1000, type=int)
    parser.add_argument("--test_for", help="Evaluate the performance for this number of episodes each time. Only during training.", default=250, type=int)
    parser.add_argument("--epsilon_start", help="Sets the start epsilon for exploration during training.", default=0.9, type=float)
    parser.add_argument("--epsilon_end", help="Sets the end epsilon for exploration during training.", default=0.1, type=float)
    parser.add_argument("--epsilon_max_episodes", help="Max episodes for epsilon to reach maximum decay", default=8000, type=int)
    parser.add_argument("--gamma", help="Sets gamma discount for Q-learing during training.", default=0.9, type=float)
    parser.add_argument("--alpha", help="Sets alpha for learning rate during training.", default=0.1, type=float)
    parser.add_argument("--logdir", help="Folder to store logs", default=path.join(path.dirname(path.abspath(__file__)), "logs"))
    parser.add_argument("--models_dir", help="Folder to store models", default=path.join(path.dirname(path.abspath(__file__)), "models"))
    parser.add_argument("--previous_model", help="Load the previous model. If training, it will start from here. If testing, will use to test.", type=str)
    parser.add_argument("--testing", help="Test the agent. No train.", default=False, type=bool)
    parser.add_argument("--experiment_id", help="Id of the experiment to record into Wandb.", default='', type=str)
    # Logging platform selection arguments
    parser.add_argument("--disable_wandb", help="Disable Wandb logging (enabled by default).", action='store_true')
    # Wandb-specific arguments
    parser.add_argument("--wandb_project", help="Wandb project name.", default="netsec-conceptual-qlearning", type=str)
    parser.add_argument("--wandb_entity", help="Wandb team/user name.", default=None, type=str)
    parser.add_argument("--wandb_mode", help="Wandb logging mode (online/offline).", default="online", type=str)
    parser.add_argument("--wandb_group", help="Wandb group name for organizing runs.", default=None, type=str)
    parser.add_argument("--store_actions", help="Store actions in the log file q_agents_actions.log.", default=False, type=bool)
    parser.add_argument("--store_models_every", help="Store a model to disk every these number of episodes.", default=2000, type=int)
    parser.add_argument("--env_conf", help="Configuration file of the env. Only for logging purposes.", required=False, default='./env/netsecenv_conf.yaml', type=str)
    parser.add_argument("--early_stop_threshold", help="Threshold for win rate for testing. If the value goes over this threshold, the training is stopped. Defaults to 95 (mean 95%% perc)", required=False, default=95, type=float)
    parser.add_argument("--apm", help="Maximum actions per minute", default=1000000, type=int, required=False)
    parser.add_argument("--enhanced_logging", help="Enable enhanced concept mapping logging", default=False, action='store_true')
    args = parser.parse_args()

    # Check that the directory for the logs exist
    if not path.exists(args.logdir):
        makedirs(args.logdir)
    log_level = logging.INFO if args.enhanced_logging else logging.ERROR
    logging.basicConfig(filename=path.join(args.logdir, "conceptual_q_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=log_level)

    # Create agent object
    agent = QAgent(args.host, args.port, alpha=args.alpha, gamma=args.gamma, epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_max_episodes=args.epsilon_max_episodes, apm_limit=args.apm)
    
    if args.enhanced_logging:
        agent.enable_enhanced_logging(verbose=True)

    # Set logging platform usage based on flags
    # wandb is enabled by default
    args.use_wandb = not args.disable_wandb
    
    # Early stop flag. Used to stop the training if the win rate goes over a threshold.
    early_stop = False

    # If there is a previous model passed. Always use it for both training and testing.
    if args.previous_model:
        # Load table
        agent._logger.info(f'Loading the previous model in file {args.previous_model}')
        try:
            agent.load_q_table(args.previous_model)
        except FileNotFoundError:
            message = f'Problem loading the file: {args.previous_model}'
            agent._logger.info(message)
            print(message)


    # Set W&B for tracking
    if not args.testing:
        # Experiment name        
        experiment_name = "Training and Eval of Conceptual Q-learning Agent"
    elif args.testing:
        # Experiment name        
        experiment_name = "Testing of Conceptual Q-learning Agent against defender agent"

    # This code runs for both training and testing. 
    # How ti works:
    # - Train for --episodes episodes
    # - Every --test_each 'episodes' stop training and start testin
    # - Test for --test_for episodes
    # - When each episode finishes you have: steps played, return, win/lose.
    # - For each episode, store all values and compute the avg and std of each of them
    # - Every --test_for episodes and at the end of the testing, report results in log file, remote log and console.

    # Register the agent
    # Obsservation is in IPs
    observation = agent.register()
    if not observation:
        raise Exception("Problem registering the agent")
    # Convert the obvervation to conceptual observation
    concept_observation = convert_ips_to_concepts(observation, agent._logger, agent.concept_logger)
    # From now one the observation will be in concepts

    # Start the train/eval/test loop
    try:
        # Initialize wandb if enabled
        if args.use_wandb:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                group=args.wandb_group,
                name=f"ConceptualQ-{experiment_name}.ID{args.experiment_id}",
                mode=args.wandb_mode
            )
        
        try:
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

            # Get git commit information
            netsecenv_command = "cd ..; git rev-parse HEAD"
            netsecenv_git_result = subprocess.run(netsecenv_command, shell=True, capture_output=True, text=True).stdout
            agents_command = "git rev-parse HEAD"
            agents_git_result = subprocess.run(agents_command, shell=True, capture_output=True, text=True).stdout
            agent._logger.info(f'Using commits. NetSecEnv: {netsecenv_git_result}. Agents: {agents_git_result}')
            
            # Log configuration to Wandb if enabled
            if args.use_wandb:
                wandb.config.update({
                    "alpha": args.alpha,
                    "gamma": args.gamma,
                    "epsilon_start": args.epsilon_start,
                    "epsilon_end": args.epsilon_end,
                    "epsilon_max_episodes": args.epsilon_max_episodes,
                    "episodes": args.episodes,
                    "test_each": args.test_each,
                    "test_for": args.test_for,
                    "testing": args.testing,
                    "experiment_name": experiment_name,
                    "agent_type": "conceptual_q_learning",
                    "concept_mapping": "stable_hosts",
                    "netsecenv_commit": netsecenv_git_result.strip(),
                    "agents_commit": agents_git_result.strip()
                })
                
                if args.previous_model:
                    wandb.config.update({"previous_model_loaded": str(args.previous_model)})
                
                # Log the env conf
                try:
                    if path.exists(args.env_conf):
                        wandb.save(args.env_conf, base_path=path.dirname(path.abspath(args.env_conf)))
                    else:
                        agent._logger.warning(f"Environment config file not found: {args.env_conf}")
                        wandb.config.update({"env_conf_path": args.env_conf})
                except Exception as e:
                    agent._logger.warning(f"Could not save env config file to Wandb: {e}")
                    wandb.config.update({"env_conf_path": args.env_conf})
            agent._logger.info(f'Epsilon Start: {agent.epsilon_start}')
            agent._logger.info(f'Epsilon End: {agent.epsilon_end}')
            agent._logger.info(f'Epsilon Max Episodes: {agent.epsilon_max_episodes}')

            # Start training
            for episode in range(1, args.episodes + 1):
                if not early_stop:
                    # Play 1 episode only
                    observation, num_steps = agent.play_game(concept_observation, testing=args.testing, episode_num=episode)       

                    # Do we have a good observation? It can be that it run of of actions and observation is None
                    if observation:
                        state = observation.state
                        reward = observation.reward
                        end = observation.end
                        info = observation.info

                        if observation.info and observation.info['end_reason'] == AgentStatus.Fail:
                            detected +=1
                            num_detected_steps += [num_steps]
                            num_detected_returns += [reward]
                        elif observation.info and observation.info['end_reason'] == AgentStatus.Success:
                            wins += 1
                            num_win_steps += [num_steps]
                            num_win_returns += [reward]
                        elif observation.info and observation.info['end_reason'] == AgentStatus.TimeoutReached:
                            max_steps += 1
                            num_max_steps_steps += [num_steps]
                            num_max_steps_returns += [reward]

                        if args.testing:
                            agent._logger.error(f"Testing episode {episode}: Steps={num_steps}. Reward {reward}. States in Q_table = {len(agent.q_values)}")
                        elif not args.testing:
                            agent._logger.error(f"Training episode {episode}: Steps={num_steps}. Reward {reward}. States in Q_table = {len(agent.q_values)}")

                    # Reset the game here, after we analyzed the data of the last observation.
                    # After each episode we need to reset the game 
                    observation = agent.request_game_reset()
                    # Reset the history of actions
                    agent.actions_history = set()

                    # Convert the obvervation to conceptual observation
                    concept_observation = convert_ips_to_concepts(observation, agent._logger, agent.concept_logger)
                    # From now one the observation will be in concepts

                    eval_win_rate = (wins/episode) * 100
                    eval_detection_rate = (detected/episode) * 100
                    eval_average_returns = np.mean(num_detected_returns + num_win_returns + num_max_steps_returns)
                    eval_std_returns = np.std(num_detected_returns + num_win_returns + num_max_steps_returns)
                    eval_average_episode_steps = np.mean(num_win_steps + num_detected_steps + num_max_steps_steps)
                    eval_std_episode_steps = np.std(num_win_steps + num_detected_steps + num_max_steps_steps)
                    eval_average_win_steps = np.mean(num_win_steps)
                    eval_std_win_steps = np.std(num_win_steps)
                    eval_average_detected_steps = np.mean(num_detected_steps)
                    eval_std_detected_steps = np.std(num_detected_steps)
                    eval_average_max_steps_steps = np.mean(num_max_steps_steps)
                    eval_std_max_steps_steps = np.std(num_max_steps_steps)

                    # Now Test, log and report. This happens every X training episodes
                    # If we are in training mode, we test for --test_for episodes
                    # If we are testing mode, this stop is not necessary since the model does not change as in training.
                    if episode % args.test_each == 0 and episode != 0 and not args.testing:
                        # First report performance of trained model up to here
                        text = f'''Performance after {episode} training episodes.
                            Wins={wins},
                            Detections={detected},
                            winrate={eval_win_rate:.3f}%,
                            detection_rate={eval_detection_rate:.3f}%,
                            average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
                            average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
                            average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
                            average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
                            average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
                            epsilon={agent.current_epsilon}
                            '''
                        agent._logger.info(text)
                        
                        # Log evaluation metrics to Wandb if enabled
                        if args.use_wandb:
                            wandb.log({
                                "eval_avg_win_rate": eval_win_rate,
                                "eval_avg_detection_rate": eval_detection_rate,
                                "eval_avg_returns": eval_average_returns,
                                "eval_std_returns": eval_std_returns,
                                "eval_avg_episode_steps": eval_average_episode_steps,
                                "eval_std_episode_steps": eval_std_episode_steps,
                                "eval_avg_win_steps": eval_average_win_steps,
                                "eval_std_win_steps": eval_std_win_steps,
                                "eval_avg_detected_steps": eval_average_detected_steps,
                                "eval_std_detected_steps": eval_std_detected_steps,
                                "eval_avg_max_steps_steps": eval_average_max_steps_steps,
                                "eval_std_max_steps_steps": eval_std_max_steps_steps,
                                "current_epsilon": agent.current_epsilon,
                                "current_episode": episode,
                                "q_table_size": len(agent.q_values),
                                "unique_states": len(agent._str_to_id)
                            }, step=episode)

                        # Now we need to keep statistics during the --test_for number of episodes
                        test_wins = 0
                        test_detected = 0
                        test_max_steps = 0
                        test_num_win_steps = []
                        test_num_detected_steps = []
                        test_num_max_steps_steps = []
                        test_num_detected_returns = []
                        test_num_win_returns = []
                        test_num_max_steps_returns = []

                        # Test
                        for test_episode in range(1, args.test_for + 1):
                            # Play 1 episode
                            # See that we force the model to freeze by telling it that it is in 'testing' mode.
                            # Also the episode_num is not updated since this controls the decay of the epsilon during training and we dont want to change that
                            test_observation, test_num_steps = agent.play_game(concept_observation, testing=True, episode_num=episode)       

                            # Do we have a good observation? It can be that it run of of actions and observation is None
                            if test_observation:
                                test_state = test_observation.state
                                test_reward = test_observation.reward
                                test_end = test_observation.end
                                test_info = test_observation.info

                                if test_info and test_info['end_reason'] == AgentStatus.Fail:
                                    test_detected +=1
                                    test_num_detected_steps += [num_steps]
                                    test_num_detected_returns += [reward]
                                elif test_info and test_info['end_reason'] == AgentStatus.Success:
                                    test_wins += 1
                                    test_num_win_steps += [num_steps]
                                    test_num_win_returns += [reward]
                                elif test_info and test_info['end_reason'] == AgentStatus.TimeoutReached:
                                    test_max_steps += 1
                                    test_num_max_steps_steps += [num_steps]
                                    test_num_max_steps_returns += [reward]

                                agent._logger.error(f"\tTesting episode {test_episode}: Steps={test_num_steps}. Reward {test_reward}. States in Q_table = {len(agent.q_values)}")

                            # Reset the game
                            test_observation = agent.request_game_reset()
                            # Reset the history of actions
                            agent.actions_history = set()

                            # Convert the obvervation to conceptual observation
                            test_observation = convert_ips_to_concepts(test_observation, agent._logger, agent.concept_logger)
                            # From now one the observation will be in concepts

                            test_win_rate = (test_wins/test_episode) * 100
                            test_detection_rate = (test_detected/test_episode) * 100
                            test_average_returns = np.mean(test_num_detected_returns + test_num_win_returns + test_num_max_steps_returns)
                            test_std_returns = np.std(test_num_detected_returns + test_num_win_returns + test_num_max_steps_returns)
                            test_average_episode_steps = np.mean(test_num_win_steps + test_num_detected_steps + test_num_max_steps_steps)
                            test_std_episode_steps = np.std(test_num_win_steps + test_num_detected_steps + test_num_max_steps_steps)
                            test_average_win_steps = np.mean(test_num_win_steps)
                            test_std_win_steps = np.std(test_num_win_steps)
                            test_average_detected_steps = np.mean(test_num_detected_steps)
                            test_std_detected_steps = np.std(test_num_detected_steps)
                            test_average_max_steps_steps = np.mean(test_num_max_steps_steps)
                            test_std_max_steps_steps = np.std(test_num_max_steps_steps)

                            # Store the model every --eval_each episodes. 
                            # Use episode (training counter) and not test_episode (test counter)
                            if episode % args.store_models_every == 0 and episode != 0:
                                agent.store_q_table(args.models_dir, f'conceptual_q_agent.experiment{args.experiment_id}-episodes-{episode}.pickle')

                        text = f'''Tested for {test_episode} episodes after {episode} training episode.
                            Wins={test_wins},
                            Detections={test_detected},
                            winrate={test_win_rate:.3f}%,
                            detection_rate={test_detection_rate:.3f}%,
                            average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
                            average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
                            average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
                            average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
                            average_max_steps_steps={test_std_max_steps_steps:.3f} +- {test_std_max_steps_steps:.3f},
                            epsilon={agent.current_epsilon}
                            '''
                        agent._logger.info(text)
                        print(text)

                        # Log test metrics to Wandb if enabled
                        if args.use_wandb:
                            wandb.log({
                                "test_avg_win_rate": test_win_rate,
                                "test_avg_detection_rate": test_detection_rate,
                                "test_avg_returns": test_average_returns,
                                "test_std_returns": test_std_returns,
                                "test_avg_episode_steps": test_average_episode_steps,
                                "test_std_episode_steps": test_std_episode_steps,
                                "test_avg_win_steps": test_average_win_steps,
                                "test_std_win_steps": test_std_win_steps,
                                "test_avg_detected_steps": test_average_detected_steps,
                                "test_std_detected_steps": test_std_detected_steps,
                                "test_avg_max_steps_steps": test_average_max_steps_steps,
                                "test_std_max_steps_steps": test_std_max_steps_steps,
                                "current_epsilon": agent.current_epsilon,
                                "current_episode": episode,
                                "q_table_size": len(agent.q_values),
                                "unique_states": len(agent._str_to_id)
                            }, step=episode)

                        if test_win_rate >= args.early_stop_threshold:
                            agent.logger.info(f'Early stopping. Test win rate: {test_win_rate}. Threshold: {args.early_stop_threshold}')
                            early_stop = True

            
            # Log the last final episode when it ends
            text = f'''Final model performance after {episode} episodes.
                Wins={wins},
                Detections={detected},
                winrate={eval_win_rate:.3f}%,
                detection_rate={eval_detection_rate:.3f}%,
                average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
                average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
                average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
                average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
                average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
                epsilon={agent.current_epsilon}
                '''

            agent._logger.info(text)
            print(text)
            agent._logger.error("Terminating interaction")
            agent.terminate_connection()
        
        finally:
            # Clean up logging sessions
            if args.use_wandb:
                wandb.finish()

    except KeyboardInterrupt:
        # Store the q-table
        if not args.testing:
            agent.store_q_table(args.models_dir, f'conceptual_q_agent.experiment{args.experiment_id}-episodes-{episode}.pickle')
    finally:
        # Store the q-table
        if not args.testing:
            agent.store_q_table(args.models_dir, f'conceptual_q_agent.experiment{args.experiment_id}-episodes-{episode}.pickle')