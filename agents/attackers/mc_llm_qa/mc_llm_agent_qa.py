import logging
import argparse
import numpy as np
import pandas as pd
import mlflow
import sys
import json
from mc_llm_action_planner import MCLLMActionPlanner
from os import path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from AIDojoCoordinator.game_components import Action, ActionType, Observation, AgentStatus
from NetSecGameAgents.agents.base_agent import BaseAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
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
    
    # MC CHANGE: Add argument for the Markov Chain JSON file
    parser.add_argument(
        "--markov_chain",
        type=str,
        default="transition_probabilities.json", # Default file name
        help="Path to the Markov Chain transition probabilities JSON file.",
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

    # Add max token limit similar to llm_qa
    parser.add_argument(
        "--max_tokens_limit",
        type=int,
        default=0,
        help="If cumulative total tokens exceeds this limit, stop the run. 0 = no limit.",
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
    agent = BaseAgent(args.host, args.port, "Attacker")
    
    if not args.disable_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

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
            # MC CHANGE: Log the markov chain file used
            "markov_chain": args.markov_chain,
            "max_tokens_limit": args.max_tokens_limit,
        }
        mlflow.log_params(params)
        mlflow.set_tag("agent_role", "Attacker")

    # Statistics variables, identical to the original script
    wins = 0
    detected = 0
    reach_max_steps = 0
    returns = []
    num_steps = []
    num_win_steps = []
    num_detected_steps = []
    num_actions_repeated = []
    # Track tokens used across the whole run
    total_tokens_used = 0
    
    prompt_table = []
    
    print("Registering")
    agent.register()
    print("Done")
    
    for episode in range(1, args.test_episodes + 1):
        actions_took_in_episode = []
        evaluations = []
        logger.info(f"Running episode {episode}")
        print(f"Running episode {episode}")

        observation = agent.request_game_reset()
        num_iterations = observation.info["max_steps"]
        current_state = observation.state
        
        memories = []
        total_reward = 0
        repeated_actions = 0

        if args.llm is not None:
            # MC CHANGE: Pass the path of the JSON file to the planner
            llm_query = MCLLMActionPlanner(
                model_name=args.llm,
                goal=observation.info["goal_description"],
                memory_len=args.memory_buffer,
                api_url=args.api_url,
                markov_chain_path=args.markov_chain
            )
        
        print(observation)
        for i in range(num_iterations):
            good_action = False
            is_valid, response_dict, action, tokens_used = llm_query.get_action_from_obs_react(observation, memories)
            total_tokens_used += tokens_used
            print(f"Total tokens used so far: {total_tokens_used}")

            # Enforce max token budget if requested
            if args.max_tokens_limit > 0 and total_tokens_used > args.max_tokens_limit:
                termination_message = (
                    f"CRITICAL: Total token limit of {args.max_tokens_limit} reached "
                    f"(cumulative total: {total_tokens_used}). Terminating run."
                )
                print(termination_message)
                logger.critical(termination_message)

                # Persist data collected so far
                print("Saving data from completed episodes before exiting...")
                with open("episode_data.json", "w") as json_file:
                    json.dump(prompt_table, json_file, indent=4)
                print("Data saved to episode_data.json.")

                if not args.disable_mlflow:
                    mlflow.set_tag("termination_reason", "max_tokens_limit_exceeded")
                    mlflow.log_param("termination_episode", episode)
                    mlflow.log_metric("total_tokens_at_termination", total_tokens_used)
                    mlflow.end_run("FAILED")
                    print("MLflow run terminated.")

                print("Exiting script.")
                sys.exit(1)

            if is_valid:
                if action is None or not isinstance(action, Action):
                    print(f"ERROR: Invalid action object received: {action}")
                    evaluations.append(0)
                else:
                    try:
                        observation = agent.make_step(action)
                        logger.info(f"Observation received: {observation}")
                        total_reward += observation.reward
                        
                        # MC CHANGE: Update the Markov Chain state ONLY after a valid action
                        # This happens regardless of whether the action was helpful or not
                        llm_query.update_last_action_state(action)

                        if observation.state != current_state:
                            good_action = True
                            current_state = observation.state
                            evaluations.append(8)
                        else:
                            evaluations.append(3)
                    except Exception as e:
                        print(f"ERROR in make_step: {e}")
                        evaluations.append(0)
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
                    # MC CHANGE: No state update here since action was invalid

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
                    
                    # MC CHANGE: State was already updated above when is_valid was True
                    # Both "helpful" and "not helpful" valid actions update the MC state
                    
                    if action in actions_took_in_episode:
                        repeated_actions += 1

                    actions_took_in_episode.append(action)
            except Exception:
                memories.append(
                                (response_dict.get("action"),
                                response_dict.get("parameters")),
                                "badly formated."
                )
                print("badly formated")
            
            if len(memories) > args.memory_buffer:
                memories = memories[-args.memory_buffer:]
            
            logger.info(f"Iteration: {i} Valid: {is_valid} Good: {good_action}")
            
            if observation.end or i == (num_iterations - 1):
                # This entire block is replicated from the original for identical statistics
                if i < (num_iterations - 1):
                    reason = observation.info
                else:
                    reason = {"end_reason": AgentStatus.TimeoutReached }

                steps = i
                epi_last_reward = observation.reward
                num_actions_repeated.append(repeated_actions)
                if AgentStatus.Success == reason["end_reason"]:
                    wins += 1
                    num_win_steps.append(steps)
                    evaluations[-1] = 10
                elif AgentStatus.Fail == reason["end_reason"]:
                    detected += 1
                    num_detected_steps.append(steps)
                elif AgentStatus.TimeoutReached == reason["end_reason"]:
                    reach_max_steps += 1
                    total_reward = -100
                    steps = num_iterations
                else:
                    reach_max_steps += 1
                
                returns.append(total_reward)
                num_steps.append(steps)

                if not args.disable_mlflow:
                    # Comprehensive episodic logging, identical to the original
                    mlflow.log_metric("wins", wins, step=episode)
                    mlflow.log_metric("num_steps", steps, step=episode)
                    mlflow.log_metric("return", total_reward, step=episode)
                    mlflow.log_metric("reached_max_steps", reach_max_steps, step=episode)
                    mlflow.log_metric("detected", detected, step=episode)
                    mlflow.log_metric("total_tokens_used", total_tokens_used, step=episode)
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
            "response": llm_query.get_responses(),
            "evaluation": evaluations,
            "end_reason": str(reason["end_reason"])
        }
        prompt_table.append(episode_prompt_table)
    
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
