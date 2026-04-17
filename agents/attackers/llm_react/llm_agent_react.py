"""
This module implements an agent that is using LLM as a planning agent
Authors:  Maria Rigaki - maria.rigaki@aic.fel.cvut.cz
          Harpo MAxx - harpomaxx@gmail.com
"""
import logging
import argparse
import numpy as np
import pandas as pd
import wandb
import weave
import os
import sys
import json
import signal
from collections import Counter
from llm_action_planner import LLMActionPlanner
from os import path
from dotenv import load_dotenv, find_dotenv

from netsecgame.game_components import AgentStatus, Action, ActionType
from NetSecGameAgents.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Terminal colors (no extra dependencies)
# ---------------------------------------------------------------------------
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE    = "\033[34m"
    DIM     = "\033[2m"


def _fmt_action(response_dict):
    """Return a compact colored action string from a response dict."""
    action = (response_dict.get("action") or "?") if response_dict else "?"
    params = response_dict.get("parameters") or {} if response_dict else {}
    src  = params.get("source_host", "")
    tgt  = params.get("target_host", params.get("target_network", ""))
    detail = f"{src} → {tgt}" if src and tgt else (src or tgt or "")
    return f"{C.CYAN}{action:<18}{C.RESET}{C.DIM}{detail}{C.RESET}"


def _print_iter(episode, i, num_iterations, response_dict, is_valid, good_action, retried=False):
    retry_tag = f" {C.MAGENTA}[retry]{C.RESET}" if retried else ""
    if not is_valid:
        outcome = f"{C.RED}✗ invalid{C.RESET}"
    elif good_action:
        outcome = f"{C.GREEN}✓ helpful{C.RESET}"
    else:
        outcome = f"{C.YELLOW}~ no change{C.RESET}"
    action_str = _fmt_action(response_dict)
    print(f"  Ep {C.BOLD}{episode}{C.RESET} | {i+1:>3}/{num_iterations} | {action_str}{retry_tag} | {outcome}")


if __name__ == "__main__":
    # Write PID file so external tools can reliably kill this process
    _pid = os.getpid()
    _pid_file = f"/tmp/nsg_agent_llm_react_{_pid}.pid"
    with open(_pid_file, "w") as _f:
        _f.write(str(_pid))

    # Load environment defaults (supports both local .env and inherited env)
    try:
        _THIS_DIR = path.dirname(path.abspath(__file__))
        load_dotenv(path.join(_THIS_DIR, ".env"), override=False)
        load_dotenv(find_dotenv(), override=False)
    except Exception:
        pass

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
        default=9000,
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
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["none", "low", "medium", "high"],
        help="Set reasoning_effort for thinking models (e.g. 'none' to disable thinking on Ollama). Omit for non-thinking models."
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
        "--max_tokens_limit",
        type=int,
        default=0,
        help="If cumulative tokens across all episodes exceed this limit, terminate the run. 0 disables the limit.",
    )

    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable W&B logging (enabled by default)",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="netsec-llm-react",
        help="W&B project name",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B team/user name",
    )

    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        help="W&B logging mode (online/offline)",
    )

    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="W&B group name for organizing runs",
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

    args.use_wandb = not args.disable_wandb

    if args.use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"llm-react-{args.llm}",
            mode=args.wandb_mode,
        )
        wandb.config.update({
            "model": args.llm,
            "memory_len": args.memory_buffer,
            "episodes": args.test_episodes,
            "host": args.host,
            "port": args.port,
            "api_url": args.api_url,
            "max_tokens_limit": args.max_tokens_limit,
            "agent_role": "Attacker",
        })
        weave.init(f"{args.wandb_entity}/{args.wandb_project}")
        logging.getLogger("weave").setLevel(logging.WARNING)

    # Run multiple episodes to compute statistics
    wins = 0
    detected = 0
    reach_max_steps = 0
    total_hard_blocked_counts = Counter()  # cross-episode breakdown
    returns = []
    num_steps = []
    num_win_steps = []
    num_detected_steps = []
    num_actions_repeated = []
    reward_memory = ""
    total_tokens_used = 0

    # Create an empty list for storing prompts, responses, and evaluations
    prompt_table = []

    def _register_interrupt_saver(get_data_fn, filename: str = "episode_data.json") -> None:
        """Register Ctrl-C handler to persist current episodes JSON before exiting."""
        def _handler(signum, frame):
            try:
                data = get_data_fn()
                with open(filename, "w") as json_file:
                    json.dump(data, json_file, indent=4)
                print(f"\nInterrupted (Ctrl-C). Saved {len(data)} episodes to {filename}.")
            except Exception as e:
                print(f"\nInterrupted (Ctrl-C). Failed to save data: {e}")
            finally:
                try:
                    if args.use_wandb:
                        wandb.log({"total_tokens_at_termination": total_tokens_used})
                        wandb.finish(exit_code=1)
                except Exception:
                    pass
                os._exit(130)
        signal.signal(signal.SIGINT, _handler)
        try:
            signal.signal(signal.SIGTERM, _handler)
        except Exception:
            pass

    _register_interrupt_saver(lambda: prompt_table)

    # We are still not using this, but we keep track
    is_detected = False

    # Initialize the game
    print(f"{C.DIM}Registering agent...{C.RESET}", end=" ", flush=True)
    agent.register()
    print(f"{C.GREEN}ready{C.RESET}")

    for episode in range(1, args.test_episodes + 1):
        actions_took_in_episode = []
        evaluations = []  # used for prompt table storage.
        logger.info(f"Running episode {episode}")
        print(f"\n{C.BOLD}{C.BLUE}Episode {episode}/{args.test_episodes}{C.RESET}")

        # Reset the game at every episode and store the goal that changes
        observation = agent.request_game_reset()
        num_iterations = observation.info["max_steps"]
        current_state = observation.state

        taken_action = None
        memories = []
        total_reward = 0
        num_actions = 0
        repeated_actions = 0
        tried_in_state = {}       # dict[str, set[Action]] — per-state blacklist
        tried_globally = set()    # set[Action] — cross-state blacklist for idempotent actions
        hard_blocked_actions = 0  # LLM proposed a forbidden action despite the prompt
        hard_blocked_counts = Counter()  # action_type -> hard-block count

        if args.llm is not None:
            llm_query = LLMActionPlanner(
                model_name=args.llm,
                goal=observation.info["goal_description"],
                memory_len=args.memory_buffer,
                api_url=args.api_url,
                reasoning_effort=args.reasoning_effort,
                use_reflection=args.use_reflection,
                use_self_consistency=args.use_self_consistency
            )
        print(f"  {C.DIM}Goal: {observation.info.get('goal_description','?')} | Max steps: {num_iterations}{C.RESET}")

        for i in range(num_iterations):
            good_action = False
            pre_exec_state_key = str(current_state)
            forbidden = tried_in_state.get(pre_exec_state_key, set()) | tried_globally
            result = llm_query.get_action_from_obs_react(observation, memories, forbidden_actions=forbidden)

            # Handle 3, 4, or 5-value returns for backward compatibility
            if len(result) == 5:
                is_valid, response_dict, action, tokens_used, retried = result
            elif len(result) == 4:
                is_valid, response_dict, action, retried = result
                tokens_used = 0
            else:
                is_valid, response_dict, action = result
                tokens_used = 0
                retried = False

            # Update and enforce token limit if configured
            try:
                total_tokens_used += int(tokens_used or 0)
            except Exception:
                pass
            if args.max_tokens_limit > 0 and total_tokens_used > args.max_tokens_limit:
                termination_message = (
                    f"CRITICAL: Total token limit of {args.max_tokens_limit} reached "
                    f"(cumulative total: {total_tokens_used}). Terminating run."
                )
                print(f"  {C.RED}{C.BOLD}{termination_message}{C.RESET}")
                logger.critical(termination_message)
                try:
                    with open("episode_data.json", "w") as json_file:
                        json.dump(prompt_table, json_file, indent=4)
                except Exception:
                    pass
                if args.use_wandb:
                    try:
                        wandb.log({
                            "termination_episode": episode,
                            "total_tokens_at_termination": total_tokens_used,
                        })
                        wandb.finish(exit_code=1)
                    except Exception:
                        pass
                sys.exit(1)

            # Hard-block: reject if LLM returned a forbidden action despite the prompt
            if is_valid and action is not None and (
                action in tried_in_state.get(pre_exec_state_key, set()) or action in tried_globally
            ):
                hard_blocked_actions += 1
                hard_blocked_counts[response_dict.get("action", "unknown")] += 1
                is_valid = False
                action = None
                memories.append(
                    (
                        (response_dict["action"], response_dict["parameters"]),
                        "already tried in this exact state — choose a DIFFERENT action."
                    )
                )
                logger.info(f"Hard-blocked: LLM ignored forbidden prompt for action in state {pre_exec_state_key}")
                print(f"  {C.MAGENTA}⊘ hard-blocked: {response_dict.get('action','?')} already tried{C.RESET}")

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
                evaluations.append(0)

            _print_iter(episode, i, num_iterations, response_dict, is_valid, good_action, retried=retried)

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

                    # Update blacklists
                    if action is not None:
                        if pre_exec_state_key not in tried_in_state:
                            tried_in_state[pre_exec_state_key] = set()
                        tried_in_state[pre_exec_state_key].add(action)
                        # Globally forbidden once tried: these actions are idempotent
                        if action.action_type in (ActionType.ExfiltrateData, ActionType.ScanNetwork):
                            tried_globally.add(action)
            except Exception:
                # if the LLM sends a response that is not properly formatted.
                memories.append(
                    (
                        (response_dict["action"],
                         response_dict["parameters"]),
                        "badly formatted."
                    )
                )
                print(f"  {C.RED}badly formatted response{C.RESET}")

            if len(memories) > args.memory_buffer:
                # If the memory is full, remove the oldest memory
                memories.pop(0)

            logger.info(f"Iteration: {i} Valid: {is_valid} Good: {good_action} Tokens: {tokens_used}")

            if observation.end or i == (num_iterations - 1):
                if i < (num_iterations - 1):
                    reason = observation.info
                else:
                    reason = {"end_reason": AgentStatus.TimeoutReached}

                win = 0
                steps = i
                epi_last_reward = observation.reward
                num_actions_repeated += [repeated_actions]
                if AgentStatus.Success == reason["end_reason"]:
                    wins += 1
                    num_win_steps += [steps]
                    type_of_end = "win"
                    evaluations[-1] = 10
                elif AgentStatus.Fail == reason["end_reason"]:
                    detected += 1
                    num_detected_steps += [steps]
                    type_of_end = "detection"
                elif AgentStatus.TimeoutReached == reason["end_reason"]:
                    reach_max_steps += 1
                    type_of_end = "max_iterations"
                    total_reward = -100
                    steps = num_iterations
                else:
                    reach_max_steps += 1
                    type_of_end = "max_steps"
                returns += [total_reward]
                num_steps += [steps]

                if args.use_wandb:
                    wandb.log({
                        # Episodic values
                        "wins": wins,
                        "num_steps": steps,
                        "return": total_reward,
                        # Running metrics
                        "reached_max_steps": reach_max_steps,
                        "detected": detected,
                        "total_tokens_used": total_tokens_used,
                        "hard_blocked_actions": hard_blocked_actions,
                        # Running averages
                        "win_rate": (wins / episode) * 100,
                        "avg_returns": np.mean(returns),
                        "avg_steps": np.mean(num_steps),
                    }, step=episode)

                logger.info(
                    f"\tEpisode {episode} of game ended after {steps} steps. Reason: {reason}. Last reward: {epi_last_reward}"
                )
                logger.info(f"\tEpisode {episode}: hard_blocked_actions={hard_blocked_actions}, hard_blocked_by_type={dict(hard_blocked_counts)}")
                end_color = C.GREEN if type_of_end == "win" else C.RED if type_of_end == "detection" else C.YELLOW
                print(f"  {C.BOLD}── Episode {episode} ended{C.RESET} | {end_color}{type_of_end}{C.RESET} | steps={steps} | reward={epi_last_reward:.2f} | hard-blocked={hard_blocked_actions}")
                break

        total_hard_blocked_counts.update(hard_blocked_counts)

        episode_prompt_table = {
            "episode": episode,
            "state": llm_query.get_states(),
            "prompt": llm_query.get_prompts(),
            "response": llm_query.get_responses(),
            "evaluation": evaluations,
            "hard_blocked_actions": hard_blocked_actions,
            "end_reason": str(reason["end_reason"])
        }
        prompt_table.append(episode_prompt_table)

    # Save the JSON file
    with open("episode_data.json", "w") as json_file:
        json.dump(prompt_table, json_file, indent=4)

    # After all episodes are done. Compute statistics
    test_win_rate = (wins / args.test_episodes) * 100
    test_detection_rate = (detected / args.test_episodes) * 100
    test_max_steps_rate = (reach_max_steps / args.test_episodes) * 100
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

    tensorboard_dict.update({
        f"hard_blocked/{k}": v for k, v in total_hard_blocked_counts.items()
    })

    if args.use_wandb:
        wandb.log(tensorboard_dict)

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
        average_repeated_steps={test_average_repeated_steps:.3f} += {test_std_repeated_steps:.3f}
        hard_blocked_by_type={dict(total_hard_blocked_counts)}"""

    print(f"\n{C.BOLD}{C.BLUE}{'─'*60}{C.RESET}")
    print(f"{C.BOLD}{text}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'─'*60}{C.RESET}")
    logger.info(text)

    # Ensure resources are closed so the process can exit cleanly
    try:
        agent.terminate_connection()
    except Exception:
        pass

    if args.use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to finish W&B run cleanly: {e}")

    sys.exit(0)
