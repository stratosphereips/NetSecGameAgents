import sys
import os
import logging
import argparse
import numpy as np
import mlflow  # used for evaluation logging (if needed)
import random
import json
from os import path, makedirs


from AIDojoCoordinator.game_components import Action, Observation, AgentStatus, ActionType, AgentRole

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__) ))))

# with the path fixed, we can import now
from base_agent import BaseAgent
from agent_utils import generate_valid_actions


class MarkovChainAgent(BaseAgent):
    def __init__(self, host, port, role, episodes) -> None:
        super().__init__(host, port, role)
        np.set_printoptions(suppress=True, precision=6)
        self.parsed_solutions = []
        self.episodes = episodes

        # Set up a logger if not already configured by BaseAgent.
        self._logger = logging.getLogger(self.__class__.__name__)

        # Load and process the transition probabilities.
        self.transitions = self.load_and_prepare_transitions("transition_probabilities.json")

    @staticmethod
    def parse_action(action: Action) -> dict:
        return {
            "type": action.action_type.name,
            "params": action.parameters  # Adjust according to the Action's internal attribute, if needed.
        }

    @staticmethod
    def normalize_probabilities(transitions: dict) -> dict:
        """Normalize transition probabilities to ensure they sum up to 1."""
        normalized = {}
        for key, probs in transitions.items():
            total = sum(probs)
            if total != 0:
                normalized[key] = [p / total for p in probs]
            else:
                normalized[key] = probs
        return normalized

    def load_and_prepare_transitions(self, filename: str) -> dict:
        """Load transition probabilities from a JSON file and build the mapping."""
        with open(filename, "r") as file:
            transitions_data = json.load(file)

        transitions = {}
        action_mapping = {
            "ScanNetwork": ActionType.ScanNetwork,
            "FindServices": ActionType.FindServices,
            "ExploitService": ActionType.ExploitService,
            "FindData": ActionType.FindData,
            "ExfiltrateData": ActionType.ExfiltrateData,
        }

        for action_data in transitions_data["transition_probabilities"]:
            action = action_data["Action"]
            probabilities = [
                action_data["ScanNetwork"],
                action_data["FindServices"],
                action_data["ExploitService"],
                action_data["FindData"],
                action_data["ExfiltrateData"]
            ]
            if action == "Initial Action":
                transitions["Initial"] = probabilities
            else:
                transitions[action_mapping[action]] = probabilities

        return self.normalize_probabilities(transitions)

    def generate_valid_actions_separated(self, state) -> list:
        """
        Generate a list (of lists) of valid actions for each action type.
        The order of the list elements must correspond to the order of probabilities in the transitions.
        """
        valid_scan_network = set()
        valid_find_services = set()
        valid_exploit_service = set()
        valid_find_data = set()
        valid_exfiltrate_data = set()

        # Build valid actions assuming state contains the necessary attributes.
        for src_host in state.controlled_hosts:
            for network in state.known_networks:
                valid_scan_network.add(
                    Action(ActionType.ScanNetwork, {"target_network": network, "source_host": src_host})
                )
            for host in state.known_hosts:
                valid_find_services.add(
                    Action(ActionType.FindServices, {"target_host": host, "source_host": src_host})
                )
            for host, service_list in state.known_services.items():
                for service in service_list:
                    valid_exploit_service.add(
                        Action(ActionType.ExploitService,
                               {"target_host": host, "target_service": service, "source_host": src_host})
                    )
        for host in state.controlled_hosts:
            valid_find_data.add(
                Action(ActionType.FindData, {"target_host": host, "source_host": host})
            )
        for src_host, data_list in state.known_data.items():
            for data in data_list:
                for trg_host in state.controlled_hosts:
                    if trg_host != src_host:
                        valid_exfiltrate_data.add(
                            Action(ActionType.ExfiltrateData,
                                   {"target_host": trg_host, "source_host": src_host, "data": data})
                        )

        # Return valid actions in the following fixed order.
        return [
            list(valid_scan_network),
            list(valid_find_services),
            list(valid_exploit_service),
            list(valid_find_data),
            list(valid_exfiltrate_data)
        ]

    def select_action_markov_chain_agent(self, observation: Observation, last_action_type) -> Action:
        valid_actions = self.generate_valid_actions_separated(observation.state)

        # For the initial step, use the key "Initial".
        key = "Initial" if last_action_type is None else last_action_type

        if key not in self.transitions:
            raise ValueError(f"Transition probabilities for key {key} not found.")

        probabilities = self.transitions[key]
        if len(valid_actions) != len(probabilities):
            raise ValueError(f"Mismatch between number of action groups ({len(valid_actions)}) and "
                             f"provided probabilities ({len(probabilities)}) for {key}.")

        selected_action = None
        while selected_action is None:
            # Select one of the action groups based on the probabilities.
            selected_index = np.random.choice(len(valid_actions), p=probabilities)
            action_list = valid_actions[selected_index]
            if action_list:
                selected_action = random.choice(action_list)
        return selected_action

    def analyze_action(self, action: Action, current_state, new_state, observation: Observation, is_last_action=False) -> int:
        """
        Analyze a single action and return its result value.
         +1 if the action resulted in a state change,
         0 if the action was valid but did not change the state,
         -1 if the action was ineffective.
         A bonus or penalty is applied when the episode ends.
        """
        valid_actions = generate_valid_actions(current_state)

        if current_state != new_state:
            result = 1  # good action
        else:
            if action in valid_actions:
                result = 0  # valid but no state change
            else:
                result = -1  # ineffective action

        # End-of-episode bonus/penalty: use AgentStatus conversion.
        if is_last_action and observation.info and observation.info.get("end_reason"):
            agent_status_end = AgentStatus.from_string(observation.info["end_reason"])
            if agent_status_end == AgentStatus.Success:
                result = 9
            elif agent_status_end == AgentStatus.Fail:
                result = -9
            elif agent_status_end == AgentStatus.TimeoutReached:
                result = -5

        return result

    def play_game(self, observation: Observation, num_episodes: int = 1):
        """
        Returns the final observation and the total number of steps.
        """
        returns = []
        total_steps = 0

        # Evaluation counters.
        wins = 0
        detected = 0
        max_steps_count = 0
        num_win_steps = []
        num_detected_steps = []
        num_max_steps_steps = []
        num_win_returns = []
        num_detected_returns = []
        num_max_steps_returns = []

        for episode in range(num_episodes):
            self._logger.info(f"Starting episode {episode}")
            episodic_return = 0
            num_steps = 0
            current_solution = []
            last_action_type = None

            observation = self.request_game_reset()
            current_state = observation.state

            while observation and not observation.end:
                num_steps += 1
                action = self.select_action_markov_chain_agent(observation, last_action_type)
                last_action_type = action.action_type  # Using ActionType as the key for transitions

                previous_state = current_state
                observation = self.make_step(action)
                current_state = observation.state if observation else None

                is_last_action = bool(observation.end) if observation else True
                result = self.analyze_action(action, previous_state, current_state, observation, is_last_action)
                current_solution.append([self.parse_action(action), result])
                episodic_return += result

                if is_last_action:
                    self.parsed_solutions.append(current_solution)
                    break

            returns.append(episodic_return)
            total_steps += num_steps
            self._logger.info(f"Episode {episode} ended in {num_steps} steps with return {episodic_return}. "
                              f"Mean return so far: {np.mean(returns):.3f} ± {np.std(returns):.3f}")

            # --- End-of-episode evaluation logic ---
            if observation.info and observation.info.get('end_reason'):
                agent_status_end = AgentStatus.from_string(observation.info['end_reason'])
            else:
                agent_status_end = None

            reward = observation.reward if observation else 0

            if agent_status_end == AgentStatus.Fail:
                detected += 1
                num_detected_steps += [num_steps]
                num_detected_returns += [reward]
            elif agent_status_end == AgentStatus.Success:
                wins += 1
                num_win_steps += [num_steps]
                num_win_returns += [reward]
            elif observation.info and observation.info.get('end_reason') == AgentStatus.TimeoutReached:
                max_steps_count += 1
                num_max_steps_steps += [num_steps]
                num_max_steps_returns += [reward]

        self._logger.info(f"Final results after {num_episodes} episodes: "
                          f"Mean return = {np.mean(returns):.3f} ± {np.std(returns):.3f}, "
                          f"Total steps = {total_steps}")

        # Optionally, save the parsed solutions.
        self.save_solutions_json(os.path.join("results", "parsed_population.json"))

        return observation, total_steps

    def save_solutions_json(self, filepath: str):
        """Save parsed solutions into a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                try:
                    outer_array = json.load(f)
                except json.JSONDecodeError:
                    outer_array = []
        else:
            outer_array = []

        parsed_solutions_run = []
        for solution in self.parsed_solutions:
            solution_str_list = []
            for action_result in solution:
                if isinstance(action_result, list) and len(action_result) == 2:
                    action_str = f"[Action {str(action_result[0])}, {str(action_result[1])}]"
                    solution_str_list.append(action_str)
                else:
                    solution_str_list.append(str(action_result))
            parsed_solutions_run.append(solution_str_list)

        outer_array.append(parsed_solutions_run)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(outer_array, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, required=False)
    parser.add_argument("--episodes", help="Number of episodes to play", default=100, type=int)
    parser.add_argument("--logdir", help="Folder to store logs",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--evaluate", help="Evaluate the agent performance", action="store_true")
    parser.add_argument("--mlflow_url", help="URL for mlflow tracking server", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        makedirs(args.logdir)
    logging.basicConfig(
        filename=os.path.join(args.logdir, "genetic_agent.log"),
        filemode='w',
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    agent = MarkovChainAgent(args.host, args.port, AgentRole.Attacker, args.episodes)
    observation = agent.register()

    if not args.evaluate:
        observation, total_steps = agent.play_game(observation, args.episodes)
        agent._logger.info("Terminating interaction")
        agent.terminate_connection()
    else:
        experiment_name = "Evaluation of Genetic Agent"
        if args.mlflow_url:
            mlflow.set_tracking_uri(args.mlflow_url)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=experiment_name) as run:
            wins = 0
            detected = 0
            max_steps_count = 0
            num_win_steps = []
            num_detected_steps = []
            num_max_steps_steps = []
            num_win_returns = []
            num_detected_returns = []
            num_max_steps_returns = []

            mlflow.set_tag("experiment_name", experiment_name)
            mlflow.set_tag("notes", "Evaluation of Genetic Agent")
            mlflow.set_tag("episode_number", args.episodes)

            for episode in range(1, args.episodes + 1):
                agent._logger.info(f"Starting evaluation episode {episode}")
                observation, num_steps = agent.play_game(observation, 1)

                if observation.info and observation.info.get('end_reason'):
                    agent_status_end = AgentStatus.from_string(observation.info['end_reason'])
                else:
                    agent_status_end = None

                reward = observation.reward if observation else 0

                if agent_status_end == AgentStatus.Fail:
                    detected += 1
                    num_detected_steps += [num_steps]
                    num_detected_returns += [reward]
                elif agent_status_end == AgentStatus.Success:
                    wins += 1
                    num_win_steps += [num_steps]
                    num_win_returns += [reward]
                elif observation.info and observation.info.get('end_reason') == AgentStatus.TimeoutReached:
                    max_steps_count += 1
                    num_max_steps_steps += [num_steps]
                    num_max_steps_returns += [reward]

                observation = agent.request_game_reset()

                eval_win_rate = (wins / episode) * 100
                eval_detection_rate = (detected / episode) * 100
                all_returns = num_win_returns + num_detected_returns + num_max_steps_returns
                eval_average_returns = np.mean(all_returns) if all_returns else 0
                eval_std_returns = np.std(all_returns) if all_returns else 0

                if episode % 10 == 0:
                    text = (f"After {episode} episodes: Wins={wins}, Detections={detected}, "
                            f"Winrate={eval_win_rate:.3f}%, Detection Rate={eval_detection_rate:.3f}%, "
                            f"Average Returns={eval_average_returns:.3f} ± {eval_std_returns:.3f}")
                    agent._logger.info(text)
                    mlflow.log_metric("eval_win_rate", eval_win_rate, step=episode)
                    mlflow.log_metric("eval_detection_rate", eval_detection_rate, step=episode)
                    mlflow.log_metric("eval_average_returns", eval_average_returns, step=episode)
                    mlflow.log_metric("eval_std_returns", eval_std_returns, step=episode)

            agent._logger.info("Terminating evaluation interaction")
            agent.terminate_connection()
            experiment_id = run.info.experiment_id
            run_id = run.info.run_id
            storage_location = "locally" if not args.mlflow_url else f"at {args.mlflow_url}"
            agent._logger.info(f"MLflow Experiment ID: {experiment_id}")
            agent._logger.info(f"MLflow Run ID: {run_id}")
            agent._logger.info(f"Experiment saved {storage_location}")
