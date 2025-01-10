"""
This module implements an agent that is using LLM as a planning agent 
Authors:  Maria Rigaki - maria.rigaki@aic.fel.cvut.cz
          Harpo MAxx - harpomaxx@gmail.com
"""
import logging
import argparse
from llm_action_planner import LLMActionPlanner
import logging
import numpy as np
import pandas as pd
import mlflow

from env.game_components import ActionType
from base_agent import BaseAgent

mlflow.set_tracking_uri("http://147.32.83.60")
mlflow.set_experiment("LLM_QA_netsecgame_dec2024")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
       # choices=[
       #     "gpt-4",
       #     "gpt-4-turbo-preview",
       #     "gpt-3.5-turbo",
       #     "gpt-3.5-turbo-16k",
       #     "HuggingFaceH4/zephyr-7b-beta",
       # ],
        default="gpt-3.5-turbo",
        help="LLM used with OpenAI API",
    )
    parser.add_argument(
        "--test_episodes",
        help="Number of test episodes to run",
        default=30,
        action="store",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--memory_buffer",
        help="Number of actions to remember and pass to the LLM",
        default=5,
        action="store",
        required=False,
        type=int,
    )
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
    args = parser.parse_args()

    logging.basicConfig(
        filename="llm_qa.log",
        filemode="w",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger("llm_qa")
    logger.info("Start")
    agent = BaseAgent(args.host, args.port, "Attacker")
    experiment_description = "LLM_QA_netsecgame_dec2024." + f"Model: {args.llm}"
    mlflow.start_run(description=experiment_description)

    params = {
        "model": args.llm,
        "memory_len": args.memory_buffer,
        "episodes": args.test_episodes,
    }
    mlflow.log_params(params)

   

    # Run multiple episodes to compute statistics
    wins = 0
    detected = 0
    reach_max_steps = 0
    returns = []
    num_steps = []
    num_win_steps = []
    num_detected_steps = []
    num_actions_repeated = []
    reward_memory = ""

 
    # Create an empty DataFrame for storing prompts and responses, and evaluations
    prompt_table = pd.DataFrame(columns=["state", "prompt", "response", "evaluation"])
    
    
    # We are still not using this, but we keep track
    is_detected = False

    # Initialize the game
    print("Registering")
    agent.register()
    print("Done")
    for episode in range(1, args.test_episodes + 1):
        actions_took_in_episode = []
        evaluations = [] # used for prompt table storage.
        logger.info(f"Running episode {episode}")
        print(f"Running episode {episode}")

        # Reset the game at every episode and store the goal that changes
        observation = agent.request_game_reset()
        num_iterations = observation.info["max_steps"]
        current_state = observation.state

        taken_action = None
        memories = []
        total_reward = 0
        num_actions = 0
        repeated_actions = 0

        if args.llm is not None:
            llm_query = LLMActionPlanner(
            model_name=args.llm,
            goal=observation.info["goal_description"],
            memory_len=args.memory_buffer
        )

        for i in range(num_iterations):
            good_action = False
            #is_json_ok = True
            is_valid, response_dict, action = llm_query.get_action_from_obs_react(observation, memories)
            if is_valid:
                observation = agent.make_step(action)
                logger.info(f"Observation received: {observation}")
                taken_action = action
                total_reward += observation.reward

                if observation.state != current_state:
                    good_action = True
                    current_state = observation.state
                    evaluations.append(8)
                else:
                    evaluations.append(3)
            else:
                print("Invalid action")
                evaluations.append(0)

            try:
                if not is_valid:
                    memories.append(
                        (
                            (response_dict["action"],
                            response_dict["parameters"]),
                            "not valid based on your status."
                        )
                    )
                else:
                    # This is based on the assumption that more valid actions in the state are better/more helpful.
                    # But we could a manual evaluation based on the prior knowledge and weight the different components.
                    # For example: finding new data is better than discovering hosts (?)
                    if good_action:
                        memories.append(
                            (
                                (response_dict["action"],
                                response_dict["parameters"]),
                                "helpful."
                            )
                        )
                    else:
                        memories.append(
                            (
                                (response_dict["action"],
                                response_dict["parameters"]),
                                "not helpful."
                            )
                        )

                    # If the action was repeated count it
                    if action in actions_took_in_episode:
                        repeated_actions += 1

                    # Store action in memory of all actions so far
                    actions_took_in_episode.append(action)
            except:
                # if the LLM sends a response that is not properly formatted.
                memories.append(
                                (response_dict["action"],
                                response_dict["parameters"]),
                                "badly formated."
                )
            
           # logger.info(f"Iteration: {i} JSON: {is_json_ok} Valid: {is_valid} Good: {good_action}")
            logger.info(f"Iteration: {i} Valid: {is_valid} Good: {good_action}")
           
            if observation.end or i == (
                num_iterations - 1
            ):  # if it is the last iteration gather statistics
                if i < (num_iterations - 1):
                    # TODO: Fix this
                    reason = observation.info
                else:
                    reason = {"end_reason": "max_iterations"}

                win = 0
                # is_detected if boolean
                # is_detected = observation.info.detected
                # TODO: Fix this
                steps = i
                epi_last_reward = observation.reward
                num_actions_repeated += [repeated_actions]
                if "goal_reached" in reason["end_reason"]:
                    wins += 1
                    num_win_steps += [steps]
                    type_of_end = "win"
                    evaluations[-1] = 10
                elif "detected" in reason["end_reason"]:
                    detected += 1
                    num_detected_steps += [steps]
                    type_of_end = "detection"
                elif "max_iterations" in reason["end_reason"]:
                    # TODO: Fix this
                    reach_max_steps += 1
                    type_of_end = "max_iterations"
                    total_reward = -100
                    #steps = observation.info["max_steps"] #this fails
                    steps = num_iterations
                else:
                    reach_max_steps += 1
                    type_of_end = "max_steps"
                returns += [total_reward]
                num_steps += [steps]

                # Episodic value
                mlflow.log_metric("wins", wins, step=episode)
                mlflow.log_metric("num_steps", steps, step=episode)
                mlflow.log_metric("return", total_reward, step=episode)

                # Running metrics
                mlflow.log_metric("wins", wins, step=episode)
                mlflow.log_metric("reached_max_steps", reach_max_steps, step=episode)
                mlflow.log_metric("detected", detected, step=episode)

                # Running averages
                mlflow.log_metric("win_rate", (wins / (episode)) * 100, step=episode)
                mlflow.log_metric("avg_returns", np.mean(returns), step=episode)
                mlflow.log_metric("avg_steps", np.mean(num_steps), step=episode)

                logger.info(
                    f"\tEpisode {episode} of game ended after {steps} steps. Reason: {reason}. Last reward: {epi_last_reward}"
                )
                print(
                    f"\tEpisode {episode} of game ended after {steps} steps. Reason: {reason}. Last reward: {epi_last_reward}"
                )
                break

        episode_prompt_table = {
            "state": llm_query.get_states(),
            "prompt": llm_query.get_prompts(),
            "response": llm_query.get_responses(),
            "evaluation": evaluations,
        }
        episode_prompt_table = pd.DataFrame(episode_prompt_table)
        prompt_table = pd.concat([prompt_table,episode_prompt_table],axis=0,ignore_index=True)
        
    prompt_table.to_csv("states_prompts_responses_new.csv", index=False)

    # After all episodes are done. Compute statistics
    test_win_rate = (wins / (args.test_episodes)) * 100
    test_detection_rate = (detected / (args.test_episodes)) * 100
    test_max_steps_rate = (reach_max_steps / (args.test_episodes)) * 100
    test_average_returns = np.mean(returns)
    test_std_returns = np.std(returns)
    test_average_episode_steps = np.mean(num_steps)
    test_std_episode_steps = np.std(num_steps)
    test_average_win_steps = np.mean(num_win_steps)
    test_std_win_steps = np.std(num_win_steps)
    test_average_detected_steps = np.mean(num_detected_steps)
    test_std_detected_steps = np.std(num_detected_steps)
    test_average_repeated_steps = np.mean(num_actions_repeated)
    test_std_repeated_steps = np.std(num_actions_repeated)
    # Store in tensorboard
    tensorboard_dict = {
        "test_avg_win_rate": test_win_rate,
        "test_avg_detection_rate": test_detection_rate,
        "test_avg_max_steps_rate": test_max_steps_rate,
        "test_avg_returns": test_average_returns,
        "test_std_returns": test_std_returns,
        "test_avg_episode_steps": test_average_episode_steps,
        "test_std_episode_steps": test_std_episode_steps,
        "test_avg_win_steps": test_average_win_steps,
        "test_std_win_steps": test_std_win_steps,
        "test_avg_detected_steps": test_average_detected_steps,
        "test_std_detected_steps": test_std_detected_steps,
        "test_avg_repeated_steps": test_average_repeated_steps,
        "test_std_repeated_steps": test_std_repeated_steps,
    }

    mlflow.log_metrics(tensorboard_dict)

    text = f"""Final test after {args.test_episodes} episodes
        Wins={wins},
        Detections={detected},
        winrate={test_win_rate:.3f}%,
        detection_rate={test_detection_rate:.3f}%,
        max_steps_rate={test_max_steps_rate:.3f}%,
        average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
        average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
        average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
        average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
        average_repeated_steps={test_average_repeated_steps:.3f} += {test_std_repeated_steps:.3f}"""

    print(text)
    logger.info(text)
