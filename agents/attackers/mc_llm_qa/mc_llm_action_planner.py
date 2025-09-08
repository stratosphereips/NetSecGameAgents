# Standard library imports
import sys
from os import path
import yaml
import logging
import json # [DEBUG] Imported for pretty-printing

# Third-party imports
import jinja2
import numpy as np
from dotenv import dotenv_values
from openai import OpenAI
from tenacity import retry, stop_after_attempt

# Add parent directories dynamically
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
)
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

# Local application/library specific imports
from AIDojoCoordinator.game_components import Action, ActionType, Observation
from NetSecGameAgents.agents.llm_utils import create_action_from_response, create_status_from_state
import validate_responses


class ConfigLoader:
    """Class to handle loading YAML configurations."""
    
    @staticmethod
    def load_config(file_name: str = 'prompts.yaml') -> dict:
        possible_paths = [
            path.join(path.dirname(__file__), file_name),
            path.join(path.dirname(path.dirname(__file__)), file_name),
            path.join(path.dirname(path.dirname(path.dirname(__file__))), file_name),
        ]
        for yaml_file in possible_paths:
            if path.exists(yaml_file):
                with open(yaml_file, 'r') as file:
                    return yaml.safe_load(file)
        raise FileNotFoundError(f"{file_name} not found in expected directories.")


ACTION_TYPE_TO_STRING = {
    ActionType.ScanNetwork: "ScanNetwork",
    ActionType.FindServices: "FindServices",
    ActionType.FindData: "FindData",
    ActionType.ExfiltrateData: "ExfiltrateData",
    ActionType.ExploitService: "ExploitService",
}


class MCLLMActionPlanner:
    def __init__(self, model_name: str, goal: str, memory_len: int = 10, api_url=None, config: dict = None, markov_chain_path: str = None):
        self.model = model_name
        self.config = config or ConfigLoader.load_config()

        if "gpt" in self.model:
            env_config = dotenv_values(".env")
            self.client = OpenAI(api_key=env_config["OPENAI_API_KEY"])
        else:
            self.client = OpenAI(base_url=api_url, api_key="ollama")
        
        self.memory_len = memory_len
        self.logger = logging.getLogger("REACT-agent")
        self.update_instructions(goal.lower())
        self.prompts = []
        self.states = []
        self.responses = []

        self.markov_chain = self._load_markov_chain(markov_chain_path)
        self.last_action_state = "Initial Action"

    def _load_markov_chain(self, json_path: str) -> dict:
        if not json_path or not path.exists(json_path):
            self.logger.warning("Markov Chain JSON not found or path not provided. Planner will not use it.")
            return None
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            transition_dict = {
                item['Action']: {k: v for k, v in item.items() if k != 'Action'}
                for item in data['transition_probabilities']
            }
            self.logger.info("Successfully loaded Markov Chain.")
            #print(f"\n[SETUP] Markov Chain loaded successfully. States: {list(transition_dict.keys())}")
            return transition_dict
        except Exception as e:
            self.logger.error(f"Failed to load or parse Markov Chain JSON: {e}")
            return None

    def _select_next_action_type(self) -> str:
        if not self.markov_chain:
            return None 

        current_transitions = self.markov_chain.get(self.last_action_state)
        
        if not current_transitions:
            self.logger.warning(f"State '{self.last_action_state}' not found in Markov Chain. Defaulting to initial state.")
            current_transitions = self.markov_chain.get("Initial Action")

        #print(f"[MC-TRACE] Transition Probabilities for this state: {current_transitions}")
        
        actions = list(current_transitions.keys())
        probabilities = list(current_transitions.values())
        
        if "FinalProbability" in actions:
            final_prob_index = actions.index("FinalProbability")
            actions.pop(final_prob_index)
            probabilities.pop(final_prob_index)
        
        p_sum = sum(probabilities)
        if p_sum == 0:
            self.logger.warning(f"No valid transition probabilities from state '{self.last_action_state}'. Cannot select an action.")
            return None
            
        normalized_probabilities = [p / p_sum for p in probabilities]
        
        #print(f"[MC-TRACE] Possible Next Actions: {actions}")
        #print(f"[MC-TRACE] Normalized Probabilities: {np.round(normalized_probabilities, 3)}")
        
        next_action_type = np.random.choice(actions, p=normalized_probabilities)

        self.logger.info(f"Markov Chain state: '{self.last_action_state}'. Selected next action type: '{next_action_type}'")
        return next_action_type

    def update_last_action_state(self, last_action: Action):
        if last_action and self.markov_chain:
            action_type_str = ACTION_TYPE_TO_STRING.get(last_action.action_type)
            if action_type_str in self.markov_chain:
                self.last_action_state = action_type_str
            else:
                self.logger.warning(f"Action type '{action_type_str}' not a valid state in Markov Chain. State not updated.")


    def get_prompts(self) -> list:
        return self.prompts

    def get_responses(self) -> list:
        return self.responses
    
    def get_states(self) -> list:
        return self.states
    
    def update_instructions(self, new_goal: str) -> None:
        template = jinja2.Environment().from_string(self.config['prompts']['INSTRUCTIONS_TEMPLATE'])
        self.instructions = template.render(goal=new_goal)

    def create_mem_prompt(self, memory_list: list) -> str:
        prompt = ""
        for memory, goodness in memory_list:
            prompt += f"You have taken action {memory} in the past. This action was {goodness}.\n"
        return prompt

    @retry(stop=stop_after_attempt(3))
    def openai_query(self, msg_list: list, max_tokens: int = 60, model: str = None, fmt=None):
        llm_response = self.client.chat.completions.create(
            model=model or self.model,
            messages=msg_list,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format=fmt or {"type": "text"},
        )
        return llm_response.choices[0].message.content, llm_response.usage

    def parse_response(self, llm_response: str, state: Observation.state):
        # This method remains unchanged
        response_dict = {"action": None, "parameters": None}
        valid = False
        action = None
        try:
            validated_response, error_msg = validate_responses.validate_agent_response(llm_response)
            if validated_response is None:
                self.logger.error(f"Validation failed: {error_msg}")
                response_dict["action"] = "InvalidResponse"
                response_dict["parameters"] = {"error": error_msg, "original": llm_response}
                return valid, response_dict, action
            
            response = validated_response
            action_str = response.get("action", None)
            action_params = response.get("parameters", None)
            
            if action_str and action_params is not None:
                valid, action = create_action_from_response(response, state)
                response_dict["action"] = action_str
                response_dict["parameters"] = action_params
            else:
                self.logger.warning("Missing action or parameters in LLM response.")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            response_dict["action"] = "InvalidJSON"
            response_dict["parameters"] = llm_response
        except Exception as e:
            self.logger.error(f"Unexpected error in parse_response: {e}")
        return valid, response_dict, action

    def get_action_from_obs_react(self, observation: Observation, memory_buf: list) -> tuple:
        """
        Plans an action by combining Markov Chain guidance with multi-stage LLM reasoning.
        - Stage 0: The Markov Chain proposes an action type, which is validated by the LLM.
                   This loop continues until a viable action type with at least one target is found.
        - Stage 1: The LLM reasons about the best parameters for the validated action type,
                   listing and prioritizing options.
        - Stage 2: The LLM generates the final JSON action based on its reasoning.
        """
        self.states.append(observation.state.as_json())
        status_prompt = create_status_from_state(observation.state)
        memory_prompt = self.create_mem_prompt(memory_buf)

        # ----------- STAGE 0: PRE-VALIDATION OF ACTION TYPE -----------
        max_retries = 10 
        total_tokens_used = 0
        next_action_type_str = None
        
        for attempt in range(max_retries):
            candidate_action_type = self._select_next_action_type()
            if not candidate_action_type:
                self.logger.error("Markov Chain failed to select an action type. Cannot proceed.")
                # Ensure consistent return signature and trace logging
                self.prompts.append({"stage": "stage0", "messages": []})
                self.responses.append({"stage": "stage0", "response": "MC failed to select action type"})
                return False, {"action": "NoAction", "parameters": "MC failed"}, None, total_tokens_used
            
            self.logger.info(f"(Stage 0) Attempt {attempt + 1}/{max_retries}: Validating action type '{candidate_action_type}'...")

            q0_config = self.config['questions'][0]
            q0_template = q0_config['text']
            q0_rules = q0_config['rules']
            specific_rule = q0_rules.get(candidate_action_type)
            
            if not specific_rule:
                self.logger.error(f"No validation rule found for action type '{candidate_action_type}' in config. Retrying.")
                continue

            q0_formatted = q0_template.format(action_type=candidate_action_type, specific_rule=specific_rule)
            
            messages_stage0 = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": status_prompt}, 
                {"role": "user", "content": q0_formatted},
            ]
            
            self.logger.info(f"Stage 0 Full Prompt:\n{json.dumps(messages_stage0, indent=2)}")
            # Trace: store Stage 0 prompt
            self.prompts.append({"stage": "stage0", "messages": messages_stage0})
            try:
                response_stage0, usage0 = self.openai_query(messages_stage0, max_tokens=10)
                total_tokens_used += usage0.total_tokens
                clean_response = response_stage0.strip()
                self.logger.info(f"Stage 0 Raw Response: {clean_response}")
                num_valid_actions = int(clean_response)
                # Trace: store Stage 0 response
                self.responses.append({"stage": "stage0", "response": clean_response, "total_tokens": usage0.total_tokens})

                if num_valid_actions > 0:
                    self.logger.info(f"(Stage 0) Validation successful. Proceeding with '{candidate_action_type}'.")
                    next_action_type_str = candidate_action_type
                    break 
                else:
                    self.logger.info(f"(Stage 0) No valid targets for '{candidate_action_type}'. Retrying with new action type.")

            except (ValueError, TypeError) as e:
                self.logger.warning(f"Could not parse Stage 0 response as integer: {clean_response}. Error: {e}. Retrying.")
                # Trace: store unparsable response
                self.responses.append({"stage": "stage0", "response": clean_response, "note": "unparsable", "error": str(e)})
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during Stage 0 query: {e}. Retrying.")
                self.responses.append({"stage": "stage0", "response": "error", "error": str(e)})

        if next_action_type_str is None:
            self.logger.error(f"Failed to find a viable action type after {max_retries} attempts.")
            # Ensure consistent return signature and trace logging
            self.responses.append({"stage": "stage0", "response": "No viable action type found"})
            return False, {"action": "NoAction", "parameters": f"No viable action type found after {max_retries} retries."}, None, total_tokens_used
        
        # ----------- STAGE 1: REASONING & PRIORITIZATION (New Logic) -----------
        self.logger.info(f"(Stage 1) Generating tactical options for action type: '{next_action_type_str}'...")

        q1_template = self.config['questions'][1]['text']
        q1_formatted = q1_template.format(action_type=next_action_type_str)
        
        # Build a coherent message history for a multi-turn conversation
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": f"{status_prompt}\n{memory_prompt}"}, # Combine status and memory
            {"role": "user", "content": q1_formatted}
        ]
        
        self.logger.info(f"Stage 1 Full Prompt:\n{json.dumps(messages, indent=2)}")
        # Trace: store Stage 1 prompt
        self.prompts.append({"stage": "stage1", "messages": messages})
        
        # Query for reasoning text
        reasoning_response, usage1 = self.openai_query(messages, max_tokens=512)
        total_tokens_used += usage1.total_tokens
        self.logger.info(f"(Stage 1) Reasoning Response:\n{reasoning_response}")
        # Trace: store Stage 1 response
        self.responses.append({"stage": "stage1", "response": reasoning_response, "total_tokens": usage1.total_tokens})

        # ----------- STAGE 2: FINAL JSON GENERATION (New Logic) -----------
        self.logger.info(f"(Stage 2) Filling parameters for action type: '{next_action_type_str}'...")

        q2_template = self.config['questions'][2]['text']
        action_example = self.config['action_examples'].get(next_action_type_str, "")

        # Normalize action naming to match validator expectations
        canonical_action_name = (
            "FindServices" if next_action_type_str in ("FindServices", "ScanServices") else next_action_type_str
        )

        # The final prompt includes the example for better format adherence
        q2_final_prompt = (
            f"{q2_template}\n\n"
            f"Important: Set the JSON 'action' field exactly to '{canonical_action_name}'.\n"
            f"Here is an example of the required format:\n{action_example}"
        )
        
        # Add the LLM's own reasoning to the context before asking for the final action
        messages.append({"role": "assistant", "content": reasoning_response})
        messages.append({"role": "user", "content": q2_final_prompt})

        self.logger.info(f"Stage 2 Full Prompt:\n{json.dumps(messages, indent=2)}")
        # Trace: store Stage 2 prompt
        self.prompts.append({"stage": "stage2", "messages": messages})
        
        # Query for the final JSON object
        final_json_response, usage2 = self.openai_query(messages, max_tokens=200, fmt={"type": "json_object"})
        total_tokens_used += usage2.total_tokens
        
        self.logger.info(f"(Stage 2) Response from LLM: {final_json_response}")
        print(f"(Stage 2) Response from LLM: {final_json_response}")
        # Trace: store Stage 2 response
        self.responses.append({"stage": "stage2", "response": final_json_response, "total_tokens": usage2.total_tokens})
        
        valid, resp_dict, action = self.parse_response(final_json_response, observation.state)
        return valid, resp_dict, action, total_tokens_used
