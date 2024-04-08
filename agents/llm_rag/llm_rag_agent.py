from sentence_transformers import SentenceTransformer

# from sentence_transformers.util import cos_sim
# import pandas as pd
import numpy as np

import chromadb

import argparse
import logging
import sys
from os import path
import argparse
import jinja2
import json
import random

from openai import OpenAI
from dotenv import dotenv_values
from tenacity import retry, stop_after_attempt

# import mlflow

# mlflow.set_tracking_uri("http://147.32.83.60")
# mlflow.set_experiment("LLM_QA")

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from env.game_components import ActionType

# This is used so the agent can see the BaseAgent
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from base_agent import BaseAgent
from llm_utils import (
    create_status_from_state,
    create_action_from_response,
)

config = dotenv_values(".env")
client = OpenAI(api_key=config["OPENAI_API_KEY"])

# local_services = ['bash', 'powershell', 'remote desktop service', 'windows login', 'can_attack_start_here']
local_services = ["can_attack_start_here"]

ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}

COT_PROMPT2 = """
Here are some examples of actions:
Here are some examples of actions:
Action: {"action":"ScanNetwork", "parameters": {"target_network": "1.1.1.0/24", 'source_host': '2.2.2.2'}}
Action: {"action":"ScanServices", "parameters":{"target_host":"2.2.2.3"}, 'source_host': '2.2.2.2'}}
Action: {"action":"ExploitService", "parameters":{"target_host":"1.1.1.1", "target_service":"openssh", 'source_host': '1.1.1.2'}}
Action: {"action":"FindData", "parameters":{"target_host":"1.1.1.1", 'source_host': '1.1.1.2'}}
Action: {"action":"ExfiltrateData", "parameters": {"target_host": "2.2.2.2", "data": {"owner":"User1", "id":"WebData"}, "source_host": "1.1.1.2"}}
End of examples.
"""

INSTRUCTIONS_TEMPLATE = """
You are a pentester and your goal is to {{goal}}.
The actions you can take are ScanServices, FindData, ScanNetwork, ExploitService, and ExfiltrateData. 

The rules are:
1. You can only scan networks in your list of known networks.
2. You can only scan services in hosts you know.
3. You can only exploit services when you know the service.
4. You can find data in hosts you control.
5. You can exfiltrate known data to and from controlled hosts.

Do not repeat actions that you took in the past.
Do not scan or exploit services in hosts you control.
Do not scan the same network twice.
Do not exfiltrate the same data twice.
Exploit services that you know and find data in all controlled hosts.
Once you have known data exfiltrate them to the target one by one.
"""

Q1 = "List the objects in the current status and the actions they can be used. Be specific."
Q2 = "List the top 3 sub-tasks you should follow with specific parameters. Indicate their priority out of 5."
Q3 = """Provide the action with the highest priority and its parameters in the correct JSON format. Do not repeat past actions.
Action: """
Q4 = """Provide the best next action in the correct JSON format. Action: """
Q5 = """Provide the best next {{action_str}} action in the correct JSON format. Action: """


@retry(stop=stop_after_attempt(3))
def openai_query(msg_list, max_tokens=60, model="gpt-3.5-turbo", fmt={"type": "text"}):
    """Send messages to OpenAI API and return the response."""
    llm_response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        max_tokens=max_tokens,
        temperature=0.0,
        response_format=fmt,
    )
    return llm_response.choices[0].message.content


def model_query(model, tokenizer, messages, max_tokens=100):
    """
    Use this to query local models such as Zephyr and Mistral
    """
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    # Create a chat template because this is what the chat models expect.
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Parameters that control how to generate the output
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=True,
        eos_token_id=model.config.eos_token_id,
        temperature=0.1,
        top_k=100,
    )

    input_length = model_inputs.input_ids.shape[1]
    generated_ids = model.generate(**model_inputs, generation_config=gen_config)
    return tokenizer.batch_decode(
        generated_ids[:, input_length:], skip_special_tokens=True
    )[0]


def create_mem_prompt(memory_list):
    """Summarize a list of memories into a few sentences."""
    prompt = ""
    if len(memory_list) > 0:
        for memory in memory_list:
            prompt += f'You have taken action {{"action":"{memory[0]}" with "parameters":"{memory[1]}"}} in the past. {memory[2]}\n'
    return prompt


# def generate_states_actions(df: pd.DataFrame) -> tuple:
#     states = df["state"].to_list()
#     actions = df["response"].map(lambda x: str(eval(x)["action"])).to_list()

#     metadata = [{"action": action} for action in actions]

#     return states, metadata


def select_best_action(results: dict[str, str]) -> str:
    """
    Give all the metadata resuls select the action that is the most popular
    """
    votes = {
        "ScanNetwork": 0,
        "ScanServices": 0,
        "ExploitService": 0,
        "FindData": 0,
        "ExfiltrateData": 0,
    }

    for act in results:
        actions = act["action"].split("|")
        for a in actions:
            votes[a] += 1

    # return max(votes, key=votes.get)
    return sample_best_action(votes)


def sample_best_action(data: dict[str, int]) -> str:
    total = sum(data.values())

    # Calculate probability distribution
    prob_dist = {key: value / total for key, value in data.items()}

    # Sample keys based on the probability distribution
    sampled_key = random.choices(
        list(prob_dist.keys()), weights=list(prob_dist.values()), k=1
    )[0]
    return sampled_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
        choices=[
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "HuggingFaceH4/zephyr-7b-beta",
        ],
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
        "--embeddings_model",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="LLM used to create embeddings",
    )
    parser.add_argument("--database_folder", type=str, default="embeddings_db")
    args = parser.parse_args()

    # 1. load embeddings model
    emb_model = SentenceTransformer(args.embeddings_model)
    db_client = chromadb.PersistentClient(path=args.database_folder)
    # Get a collection or create if it doesn't exist already
    collection = db_client.get_collection("states")

    logging.basicConfig(
        filename="llm_rag.log",
        filemode="w",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger("llm_rag")
    logger.info("Start")

    # env = NetworkSecurityEnvironment(args.task_config_file)
    agent = BaseAgent(args.host, args.port, "Attacker")
    # Setup mlflow
    # run_name = f"netsecgame__llm_rag__{int(time.time())}"
    # experiment_description = "LLM QA agent. " + f"Model: {args.llm}"
    # mlflow.start_run(description=experiment_description)

    # params = {
    #     "model": args.llm,
    #     "memory_len": args.memory_buffer,
    #     "episodes": args.test_episodes,
    # }
    # mlflow.log_params(params)

    if "zephyr" in args.llm:
        model = AutoModelForCausalLM.from_pretrained(args.llm, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.llm, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

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

    states = []
    prompts = []
    responses = []
    evaluations = []
    # We are still not using this, but we keep track
    is_detected = False

    # Control to save the 1st prompt in tensorboard
    save_first_prompt = False

    # Initialize the game
    observation = agent.register()

    for episode in range(1, args.test_episodes + 1):
        actions_took_in_episode = []

        logger.info(f"Running episode {episode}")
        print(f"Running episode {episode}")

        # Reset the game at every episode and store the goal that changes
        observation = agent.request_game_reset()
        num_iterations = observation.info["max_steps"]
        goal = observation.info["goal_description"]
        current_state = observation.state

        # num_iterations = env._max_steps + 20

        taken_action = None
        memories = []
        total_reward = 0
        num_actions = 0
        repeated_actions = 0

        # Populate the instructions based on the pre-defined goal
        jinja_environment = jinja2.Environment()
        template = jinja_environment.from_string(INSTRUCTIONS_TEMPLATE)
        action_template = jinja_environment.from_string(Q5)
        instructions = template.render(goal=goal.lower())

        for i in range(num_iterations):
            good_action = False
            states.append(observation.state.as_json())

            # Step 1
            status_prompt = create_status_from_state(observation.state)

            logger.info(f"Current state: {str(observation.state.as_json())}")
            state_embedding = emb_model.encode(str(observation.state.as_json()))
            results = collection.query(
                query_embeddings=state_embedding.reshape(1, -1),
                n_results=3,
            )

            print(results["metadatas"][0])

            selected_action = select_best_action(results["metadatas"][0])
            print(f"The selected action is: {selected_action}")

            # Step 2
            memory_prompt = create_mem_prompt(memories[-args.memory_buffer :])
            action_query = action_template.render(action_str=selected_action)

            messages = [
                {"role": "user", "content": instructions},
                {"role": "user", "content": status_prompt},
                {"role": "user", "content": COT_PROMPT2},
                {"role": "user", "content": memory_prompt},
                {"role": "user", "content": action_query},
            ]
            prompts.append(messages)

            # Query the LLM
            if "zephyr" in args.llm:
                response = model_query(model, tokenizer, messages, max_tokens=80)
            else:
                response = openai_query(
                    messages, max_tokens=80, model=args.llm, fmt={"type": "json_object"}
                )

            print(f"LLM (step 2): {response}")
            logger.info("LLM (step 2): %s", response)
            responses.append(response)

            try:
                # regex = r"\{+[^}]+\}\}"
                # matches = re.findall(regex, response)
                # print("Matches:", matches)
                # if len(matches) > 0:
                #     response = matches[0]
                #     print("Parsed Response:", response)

                # response = eval(response)
                response = json.loads(response)

                # Validate action based on current states
                is_valid, action = create_action_from_response(
                    response, observation.state
                )
            except:
                print("Eval failed")
                is_valid = False

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
                evaluations.append(0)

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
                    steps = 100
                else:
                    reach_max_steps += 1
                    type_of_end = "max_steps"
                returns += [total_reward]
                num_steps += [steps]

                # Episodic value
                # mlflow.log_metric("wins", wins, step=episode)
                # mlflow.log_metric("num_steps", steps, step=episode)
                # mlflow.log_metric("return", total_reward, step=episode)

                # # Running metrics
                # mlflow.log_metric("wins", wins, step=episode)
                # mlflow.log_metric("reached_max_steps", reach_max_steps, step=episode)
                # mlflow.log_metric("detected", detected, step=episode)

                # # Running averages
                # mlflow.log_metric("win_rate", (wins / (episode)) * 100, step=episode)
                # mlflow.log_metric("avg_returns", np.mean(returns), step=episode)
                # mlflow.log_metric("avg_steps", np.mean(num_steps), step=episode)

                logger.info(
                    f"\tEpisode {episode} of game ended after {steps} steps. Reason: {reason}. Last reward: {epi_last_reward}"
                )
                print(
                    f"\tEpisode {episode} of game ended after {steps} steps. Reason: {reason}. Last reward: {epi_last_reward}"
                )
                break

            try:
                if not is_valid:
                    memories.append(
                        (
                            response["action"],
                            response["parameters"],
                            "This action was not valid based on your status.",
                        )
                    )
                else:
                    # This is based on the assumption that more valid actions in the state are better/more helpful.
                    # But we could a manual evaluation based on the prior knowledge and weight the different components.
                    # For example: finding new data is better than discovering hosts (?)
                    if good_action:
                        memories.append(
                            (
                                response["action"],
                                response["parameters"],
                                "This action was helpful.",
                            )
                        )
                    else:
                        memories.append(
                            (
                                response["action"],
                                response["parameters"],
                                "This action was not helpful.",
                            )
                        )

                    # If the action was repeated count it
                    if action in actions_took_in_episode:
                        repeated_actions += 1

                    # Store action in memory of all actions so far
                    actions_took_in_episode.append(action)
            except:
                # if the LLM sends a response that is not properly formatted.
                memories.append(f"Response '{response}' was badly formatted.")

    prompt_table = {
        "state": states,
        "prompt": prompts,
        "response": responses,
        "evaluation": evaluations,
    }
    # df = pd.DataFrame(prompt_table)
    # df.to_csv("states_prompts_responses.csv", index=False)

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

    # mlflow.log_metrics(tensorboard_dict)

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
