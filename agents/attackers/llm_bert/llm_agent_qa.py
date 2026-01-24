import logging
import argparse
import numpy as np
import pandas as pd
import mlflow
import sys
import json
from os import path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

# Import from the same directory - works when run as module or script
try:
    from .llm_action_planner import LLMActionPlanner
except ImportError:
    from llm_action_planner import LLMActionPlanner

from netsecgame.game_components import AgentStatus, Action, ActionType, AgentRole
from netsecgame import BaseAgent

#mlflow.set_tracking_uri("http://147.32.83.60")
#mlflow.set_experiment("LLM_QA_netsecgame_dec2024")


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
        default=9001,
        type=int,
        action="store",
        required=False,
    )
    
    parser.add_argument(
        "--api_url",
        type=str, 
        default="http://127.0.0.1:11434/v1/"
        )

    parser.add_argument(
        "--use_reasoning",
        action="store_true",
        help="Required for models that output reasoning using <think>...</think>."
    )

    parser.add_argument(
        "--use_reflection",
        action="store_true",
        help="To use reflection prompting technique in the LLM calls."
    )

    parser.add_argument(
        "--use_self_consistency",
        action="store_true",
        help="To use self-consistency prompting technique in the LLM calls."
    )

    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://147.32.83.60",
        help="MLflow tracking server URI (default: %(default)s)",
    )

    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="LLM_QA_netsecgame_dec2024",
        help="MLflow experiment name (default: %(default)s)",
    )

    parser.add_argument(
        "--mlflow_description",
        type=str,
        default=None,
        help="Optional description for MLflow run (default is generated)",
    )



    parser.add_argument(
        "--disable_mlflow",
        action="store_true",
        help="Disable mlflow logging",
    )
    
    # CHANGED: Help text updated to be more specific.
    parser.add_argument(
        "--max_tokens_limit",
        type=int,
        default=0,
        help="If the cumulative total tokens used across all episodes exceeds this limit, the entire run is stopped. 0 means no limit. (default: %(default)s)",
    )

    parser.add_argument(
    "--classifier_model_path",
    type=str,
    default="./bertClassifier",
    help="Path to the ModernBERT action classifier model (default: %(default)s)"
    )

    parser.add_argument(
        "--mlm_model_path", 
        type=str,
        default="./bertMLM",
        help="Path to the ModernBERT masked language model (default: %(default)s)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename="llm_react.log",
        filemode="w",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger("llm_react")
    logger.info("Start")
    agent = BaseAgent(args.host, args.port, role=AgentRole.Attacker)
    
    if not args.disable_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

        # Use custom description if given, otherwise build a default
        experiment_description = args.mlflow_description or (
            f"{args.mlflow_experiment} | Model: {args.llm}"
        )

        mlflow.start_run(description=experiment_description)

        params = {
            "model": args.llm,
            "memory_len": args.memory_buffer,
            "episodes": args.test_episodes,
            "host": args.host,
            "port": args.port,
            "api_url": args.api_url,
            "max_tokens_limit": args.max_tokens_limit,
        }
        mlflow.log_params(params)
        mlflow.set_tag("agent_role", "Attacker")

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

    # ADDED: Counter for total tokens used across the entire run.
    total_tokens_used = 0
 
    # Create an empty DataFrame for storing prompts and responses, and evaluations
    prompt_table = []
    
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
            memory_len=args.memory_buffer,
            api_url=args.api_url,
            use_reasoning=args.use_reasoning,
            use_reflection=args.use_reflection,
            use_self_consistency=args.use_self_consistency,
            classifier_model_path=args.classifier_model_path,  # Add this
            mlm_model_path=args.mlm_model_path                 # Add this
        )
        print(observation)
        for i in range(num_iterations):
            good_action = False
            
            is_valid, response_dict, action, tokens_used = llm_query.get_action_from_obs_react(observation, memories)
            
            # CHANGED: Increment the total token counter for the run
            total_tokens_used += tokens_used
            print(f"Total tokens used so far: {total_tokens_used}")

            # --- MODIFIED BEHAVIOR ---
            # If total token limit is reached, save completed episode data and terminate the entire script.
            # Corrected Line
            if args.max_tokens_limit > 0 and total_tokens_used > args.max_tokens_limit:
                termination_message = f"CRITICAL: Total token limit of {args.max_tokens_limit} reached (cumulative total: {total_tokens_used}). Terminating run."
                print(termination_message)
                logger.critical(termination_message)

                print("Saving data from completed episodes before exiting...")
                with open("episode_data.json", "w") as json_file:
                    json.dump(prompt_table, json_file, indent=4)
                print("Data saved to episode_data.json.")

                if not args.disable_mlflow:
                    mlflow.set_tag("termination_reason", "max_tokens_limit_exceeded")
                    mlflow.log_param("termination_episode", episode)
                    mlflow.log_metric("total_tokens_at_termination", total_tokens_used)
                    mlflow.end_run("FAILED") # Mark the run as failed
                    print("MLflow run terminated.")

                print("Exiting script.")
                sys.exit(1) # Exit with a non-zero status code to indicate abnormal termination

            if is_valid and action is not None:
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
                print("Invalid action: ")
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
                    print("not valid based on your status.")
                else:
                    if good_action:
                        memories.append(
                            (
                                (response_dict["action"],
                                response_dict["parameters"]),
                                "helpful."
                            )
                        )
                        print("Helpful")
                    else:
                        memories.append(
                            (
                                (response_dict["action"],
                                response_dict["parameters"]),
                                "not helpful."
                            )
                        )
                        print("Not Helpful")
                    if action in actions_took_in_episode:
                        repeated_actions += 1
                    actions_took_in_episode.append(action)
            except:
                memories.append(
                    (response_dict["action"],
                     response_dict["parameters"]),
                    "badly formated."
                )
                print("badly formated")

            if len(memories) > args.memory_buffer:
                memories.pop(0)

            logger.info(f"Iteration: {i} Valid: {is_valid} Good: {good_action}")
            
            if observation.end or i == (num_iterations - 1):
                if i < (num_iterations - 1):
                    reason = observation.info
                else:
                    reason = {"end_reason": AgentStatus.TimeoutReached}

                steps = i
                epi_last_reward = observation.reward
                num_actions_repeated += [repeated_actions]
                if AgentStatus.Success == reason["end_reason"]:
                    wins += 1
                    num_win_steps += [steps]
                    evaluations[-1] = 10
                elif AgentStatus.Fail == reason["end_reason"]:
                    detected += 1
                    num_detected_steps += [steps]
                elif AgentStatus.TimeoutReached == reason["end_reason"]:
                    reach_max_steps += 1
                    total_reward = -100
                    steps = num_iterations
                else:
                    reach_max_steps += 1
                
                returns += [total_reward]
                num_steps += [steps]

                if not args.disable_mlflow:
                    mlflow.log_metric("wins", wins, step=episode)
                    mlflow.log_metric("num_steps", steps, step=episode)
                    mlflow.log_metric("return", total_reward, step=episode)
                    mlflow.log_metric("reached_max_steps", reach_max_steps, step=episode)
                    mlflow.log_metric("detected", detected, step=episode)
                    mlflow.log_metric("total_tokens_used", total_tokens_used, step=episode) # Log total tokens each episode
                    mlflow.log_metric("win_rate", (wins / episode) * 100, step=episode)
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
            "episode": episode,
            "state": llm_query.get_states(),
            "prompt": llm_query.get_prompts(),
            "modernbert_prompts": llm_query.get_modernbert_prompts(),  
            "response": llm_query.get_responses(),
            "evaluation": evaluations,
            "end_reason": str(reason["end_reason"])
        }
        prompt_table.append(episode_prompt_table)

    # This part of the code will only be reached if the run completes all episodes
    # without hitting the token limit.
    print("All episodes completed successfully.")
    with open("episode_data.json", "w") as json_file:
        json.dump(prompt_table, json_file, indent=4)

    # After all episodes are done. Compute statistics
    completed_episodes = len(prompt_table)
    test_win_rate = (wins / completed_episodes) * 100 if completed_episodes > 0 else 0
    test_detection_rate = (detected / completed_episodes) * 100 if completed_episodes > 0 else 0
    test_max_steps_rate = (reach_max_steps / completed_episodes) * 100 if completed_episodes > 0 else 0
    test_average_returns = np.mean(returns) if returns else 0
    test_std_returns = np.std(returns) if returns else 0
    test_average_episode_steps = np.mean(num_steps) if num_steps else 0
    test_std_episode_steps = np.std(num_steps) if num_steps else 0
    test_average_win_steps = np.mean(num_win_steps) if num_win_steps else 0
    test_std_win_steps = np.std(num_win_steps) if num_win_steps else 0
    test_average_detected_steps = np.mean(num_detected_steps) if num_detected_steps else 0
    test_std_detected_steps = np.std(num_detected_steps) if num_detected_steps else 0
    test_average_repeated_steps = np.mean(num_actions_repeated) if num_actions_repeated else 0
    test_std_repeated_steps = np.std(num_actions_repeated) if num_actions_repeated else 0

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
        "final_total_tokens_used": total_tokens_used,
    }

    if not args.disable_mlflow:
        mlflow.log_metrics(tensorboard_dict)
        mlflow.end_run("FINISHED")

    text = f"""Final test after {completed_episodes} episodes
        Wins={wins},
        Detections={detected},
        Max Steps Reached={reach_max_steps},
        winrate={test_win_rate:.3f}%,
        detection_rate={test_detection_rate:.3f}%,
        max_steps_rate={test_max_steps_rate:.3f}%,
        average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
        average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
        average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
        average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f},
        average_repeated_steps={test_average_repeated_steps:.3f} += {test_std_repeated_steps:.3f},
        Total Tokens Used: {total_tokens_used}
        """

    print(text)
    logger.info(text)
