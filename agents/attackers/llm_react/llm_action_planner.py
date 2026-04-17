"""
@file llm_action_planner.py

@brief Implementation of an LLM-based action planner for reactive agent systems.

This script defines classes and methods to facilitate interaction with language models (LLMs),
manage configuration files, and parse responses from LLM queries. The core functionality includes
planning actions based on observations and memory, parsing LLM responses, and dynamically loading
configuration files using YAML. Most of the code is adapted from the original implementation in
the `assistan.py` from the `interactive_tui` agent.

@author Maria Rigaki - maria.rigaki@aic.fel.cvut.cz
@author Harpo Maxx - harpomaxx@gmail.com

@date [Date]
"""

import sys
from os import path
import yaml
import logging
import json
from dotenv import dotenv_values
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import jinja2

import re
from collections import Counter
import validate_responses

from netsecgame.game_components import ActionType, Observation
from NetSecGameAgents.agents.llm_utils import create_action_from_response, create_status_from_state


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


ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}

ACTION_TYPE_TO_STR = {v: k for k, v in ACTION_MAPPER.items()}


class LLMActionPlanner:
    def __init__(self, model_name: str, goal: str, memory_len: int = 10, api_url=None, config: dict = None, reasoning_effort: str = None, use_reflection: bool = False, use_self_consistency: bool = False):
        self.model = model_name
        self.config = config or ConfigLoader.load_config()
        self.reasoning_effort = reasoning_effort
        self.use_reflection = use_reflection
        self.use_self_consistency = use_self_consistency

        # Load API key from .env if present (try several common names)
        env_config = dotenv_values(".env")
        api_key = env_config.get("OPENAI_API_KEY") or env_config.get("OPENAI_KEY") or env_config.get("API_KEY")

        if "gpt" in self.model and api_url is None:
            # Only use OpenAI default endpoint if no custom api_url provided
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = OpenAI()
        else:
            # Use custom endpoint (even for GPT models) or non-OpenAI models
            if api_key:
                self.client = OpenAI(base_url=api_url, api_key=api_key)
            else:
                self.client = OpenAI(base_url=api_url)

        self.memory_len = memory_len
        self.logger = logging.getLogger("REACT-agent")
        self.update_instructions(goal.lower())
        self.prompts = []
        self.states = []
        self.responses = []

    def get_prompts(self) -> list:
        """
        Returns the list of prompts sent to the LLM."""
        return self.prompts

    def get_responses(self) -> list:
        """
        Returns the list of responses received from the LLM. Only Stage 2 responses are included.
        """
        return self.responses

    def get_states(self) -> list:
        """
        Returns the list of states received from the LLM. In JSON format.
        """
        return self.states

    def update_instructions(self, new_goal: str) -> None:
        template = jinja2.Environment().from_string(self.config['prompts']['INSTRUCTIONS_TEMPLATE'])
        self.instructions = template.render(goal=new_goal)

    def create_mem_prompt(self, memory_list: list) -> str:
        prompt = ""
        for memory, goodness in memory_list:
            prompt += f"You have taken action {memory} in the past. This action was {goodness}.\n"
        return prompt

    def filter_messages(self, messages: list) -> list:
        """Filter out messages with None content and fix roles for API compatibility."""
        filtered = []
        for msg in messages:
            if msg.get("content") is not None and msg.get("content").strip():
                role = "system" if any(keyword in str(msg.get("content", "")).lower()
                                      for keyword in ["instruction", "goal", "you are", "your task"]) else "user"
                filtered.append({"role": role, "content": msg["content"]})
        return filtered

    def extract_json_from_response(self, response: str) -> str:
        """Extract JSON from response that may contain prefixes like 'Action: {...}'"""
        if not response:
            return response

        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response)
        if matches:
            return matches[0].strip()

        prefixes = ["Action:", "Response:", "Output:", "Result:"]
        for prefix in prefixes:
            if prefix in response:
                json_part = response.split(prefix, 1)[1].strip()
                if json_part.startswith('{') and json_part.endswith('}'):
                    return json_part

        return response.strip()

    def fix_incomplete_json(self, response: str) -> str:
        """Try to fix JSON responses that are missing the 'action' field."""
        if not response or not response.strip():
            return response

        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "action" in parsed:
                return response
            if isinstance(parsed, dict) and "action" not in parsed:
                inferred_action = self.infer_action_from_parameters(parsed)
                if inferred_action:
                    fixed_response = {
                        "action": inferred_action,
                        "parameters": parsed
                    }
                    return json.dumps(fixed_response)
        except json.JSONDecodeError:
            pass

        return response

    def extract_action_hint(self, response: str) -> str | None:
        """Try to extract the intended action name from a malformed/plain-text response."""
        if not response:
            return None
        try:
            match = re.search(r'"action"\s*:\s*"(\w+)"', response)
            if match:
                action = match.group(1)
                if action in ACTION_MAPPER:
                    return action
        except Exception:
            pass
        for action_name in ACTION_MAPPER:
            if action_name.lower() in response.lower():
                return action_name
        return None

    def infer_action_from_parameters(self, params: dict) -> str:
        """Infer the action type based on parameter keys."""
        if not isinstance(params, dict):
            return None

        param_keys = set(params.keys())

        if {"target_host", "data", "source_host"}.issubset(param_keys):
            return "ExfiltrateData"
        elif {"target_network", "source_host"}.issubset(param_keys):
            return "ScanNetwork"
        elif {"target_host", "source_host"}.issubset(param_keys) and "service" in param_keys:
            return "ExploitService"
        elif {"target_host", "source_host"}.issubset(param_keys):
            return "ScanServices"

        return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def openai_query(self, msg_list: list, max_tokens: int = 80, model: str = None, fmt=None, temperature: float = 0.0):
        filtered_messages = self.filter_messages(msg_list)

        extra = {}
        if self.reasoning_effort is not None:
            extra["reasoning_effort"] = self.reasoning_effort

        llm_response = self.client.chat.completions.create(
            model=model or self.model,
            messages=filtered_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra or None,
        )

        content = llm_response.choices[0].message.content
        if content is None:
            reasoning_content = getattr(llm_response.choices[0].message, 'reasoning_content', None)
            if reasoning_content:
                content = reasoning_content

        if content and content.startswith('```json'):
            content = content.replace('```json\n', '').replace('\n```', '').strip()

        if content:
            content = self.extract_json_from_response(content)
            content = self.fix_incomplete_json(content)

        try:
            tokens = int(llm_response.usage.total_tokens or 0)
        except Exception:
            # Fallback: provider doesn't implement usage — estimate from char counts
            input_chars = sum(len(m.get("content", "") or "") for m in msg_list)
            output_chars = len(content or "")
            tokens = (input_chars + output_chars) // 4
        return content, tokens

    def parse_response_deprecated(self, llm_response: str, state: Observation.state):
        try:
            response = json.loads(llm_response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM response as JSON.")
            return False,llm_response, None

        try:
            action_str = response["action"]
            action_params = response["parameters"]
            valid, action = create_action_from_response(response, state)
            return valid, {action_str:action_str,action_params:action_params}, action

        except KeyError:
            return False, llm_response, None

    def parse_response(self, llm_response: str, state: Observation.state, validated_dict: dict = None):
        response_dict = {"action": None, "parameters": None}
        valid = False
        action = None

        try:
            # Use pre-validated/normalized dict when available to avoid re-parsing raw string
            response = validated_dict if validated_dict is not None else json.loads(llm_response)
            action_str = response.get("action", None)
            action_params = response.get("parameters", None)

            if action_str and action_params:
                valid, action = create_action_from_response(response, state)
                response_dict["action"] = action_str
                response_dict["parameters"] = action_params
            else:
                self.logger.warning("Missing action or parameters in LLM response.")
        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM response as JSON.")
            response_dict["action"] = "InvalidJSON"
            response_dict["parameters"] = llm_response  # Return raw response for debugging
        except KeyError:
            self.logger.error("Missing keys in LLM response.")
        except ValueError as e:
            self.logger.error("Invalid action parameters from LLM (e.g. non-IP in host field): %s", e)
            response_dict["action"] = None
            response_dict["parameters"] = {}

        return valid, response_dict, action

    def check_repetition(self, memory_list):
        repetitions = 0
        past_memories = []
        for memory, goodness in memory_list:
            if memory in past_memories:
                repetitions += 1
            past_memories.append(memory)
        return repetitions

    def get_self_consistent_response(self, messages, temp=0.4, max_tokens=1024, n=3):
        candidates = []
        sc_tokens = 0
        for _ in range(n):
            response, tok = self.openai_query(messages, temperature=temp, max_tokens=max_tokens)
            sc_tokens += tok
            candidates.append(response.strip())

        counts = Counter(candidates)
        most_common = counts.most_common(1)
        if most_common:
            self.logger.info(f"Self-consistency candidates: {counts}")
            return most_common[0][0], sc_tokens
        return candidates[0], sc_tokens

    def get_action_from_obs_react(self, observation: Observation, memory_buf: list, forbidden_actions=None) -> tuple:
        self.states.append(observation.state.as_json())
        status_prompt = create_status_from_state(observation.state)
        q1 = self.config['questions'][0]['text']
        q4 = self.config['questions'][3]['text']
        cot_prompt = self.config['prompts']['COT_PROMPT']
        memory_prompt = self.create_mem_prompt(memory_buf)

        forbidden_prompt = ""
        if forbidden_actions:
            def _action_to_json(a):
                action_name = ACTION_TYPE_TO_STR.get(a.action_type, str(a.action_type))
                params = {}
                for k, v in a.parameters.items():
                    if hasattr(v, "owner") and hasattr(v, "id"):
                        params[k] = {"owner": v.owner, "id": v.id}
                    else:
                        params[k] = str(v)
                return json.dumps({"action": action_name, "parameters": params})
            lines = "\n".join(f"- {_action_to_json(a)}" for a in forbidden_actions)
            forbidden_prompt = (
                f"[SYSTEM] The following actions are FORBIDDEN in the current state "
                f"because they were already tried. Do NOT generate them:\n{lines}"
            )

        repetitions = self.check_repetition(memory_buf)
        iteration_tokens = 0
        messages = [
            {"role": "user", "content": self.instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": memory_prompt},
            {"role": "user", "content": forbidden_prompt},
            {"role": "user", "content": q1},
        ]
        self.logger.info(f"Text sent to the LLM: {messages}")

        if self.use_self_consistency:
            response, tok = self.get_self_consistent_response(messages, temp=repetitions/9, max_tokens=1024)
        else:
            response, tok = self.openai_query(messages, max_tokens=1024)
        iteration_tokens += tok

        if self.use_reflection:
            reflection_prompt = [
                {
                    "role": "user",
                    "content": f"""
                    Instructions: {self.instructions}
                    Task: {q1}

                    Status: {status_prompt}
                    Memory: {memory_prompt}

                    Reasoning:
                    {response}

                    Is this reasoning valid given the Instructions, Status, and Memory?
                    - If YES, repeat it exactly.
                    - If NO, output the corrected reasoning only (no commentary).
                    """
                }
            ]
            response, tok = self.openai_query(reflection_prompt, max_tokens=1024)
            iteration_tokens += tok

        self.logger.info(f"(Stage 1) Response from LLM: {response}")

        messages = [
            {"role": "user", "content": self.instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": cot_prompt},
            {"role": "user", "content": response},
            {"role": "user", "content": memory_prompt},
            {"role": "user", "content": forbidden_prompt},
            {"role": "user", "content": q4},
        ]
        self.prompts.append(messages)
        response, tok = self.openai_query(messages, max_tokens=1024)
        iteration_tokens += tok

        retried = False
        validated, error_msg = validate_responses.validate_agent_response(response)
        if validated is None:
            self.logger.info(f"Invalid response format: {response} - Error: {error_msg}")
            action_hint = self.extract_action_hint(response)
            if action_hint:
                retried = True
                self.logger.info(f"Retrying with format correction for action hint: {action_hint}")
                print(f"  \033[35m↩ retry: {action_hint}\033[0m")
                correction_messages = [
                    {"role": "user", "content": self.instructions},
                    {"role": "user", "content": status_prompt},
                    {"role": "user", "content": cot_prompt},
                    {"role": "user", "content": f'You chose action "{action_hint}". Using the examples above as reference, output ONLY a valid JSON object for that action with real parameter values from the current status. Action: '},
                ]
                response, tok = self.openai_query(correction_messages, max_tokens=512)
                iteration_tokens += tok
                validated, error_msg = validate_responses.validate_agent_response(response)
                self.logger.info(f"(Stage 2 retry) Response from LLM: {response}")

            if validated is None:
                try:
                    parsed_response = json.loads(response)
                except json.JSONDecodeError:
                    parsed_response = response

                response = json.dumps({
                    "action": "InvalidResponse",
                    "parameters": {
                        "error": error_msg,
                        "original": parsed_response
                    }
                }, indent=2)

        self.responses.append(response)
        self.logger.info(f"(Stage 2) Response from LLM: {response}")
        is_valid, response_dict, action = self.parse_response(response, observation.state, validated_dict=validated)
        self.logger.info(f"Iteration token usage: {iteration_tokens}")
        return is_valid, response_dict, action, iteration_tokens, retried
