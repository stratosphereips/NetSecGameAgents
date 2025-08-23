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
from tenacity import retry, stop_after_attempt
import jinja2

import re
from collections import Counter
import validate_responses

# Add parent directories dynamically
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
)
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from AIDojoCoordinator.game_components import ActionType, Observation
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


from dotenv import load_dotenv
import os
from langfuse import get_client

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configurar las variables de entorno para Langfuse (opcional, pero claro)
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

# Inicializar el cliente de Langfuse con manejo de errores
try:
    langfuse = get_client()
    print("✅ Langfuse initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Langfuse initialization failed: {e}")
    langfuse = None


class LLMActionPlanner:
    def __init__(self, model_name: str, goal: str, memory_len: int = 10, api_url=None, config: dict = None, use_reasoning: bool = False, use_reflection: bool = False, use_self_consistency: bool = False):
        self.model = model_name
        self.config = config or ConfigLoader.load_config()
        self.use_reasoning = use_reasoning
        self.use_reflection = use_reflection
        self.use_self_consistency = use_self_consistency

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
        
        # Initialize Langfuse session tracking
        self.session_id = f"agent-session-{hash(goal)}"
        self.current_span = None

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

    @retry(stop=stop_after_attempt(3))
    def openai_query(self, msg_list: list, max_tokens: int = 60, model: str = None, fmt=None, temperature: float = 0.0, parent_span=None):
        # Track LLM generation in Langfuse using v3 API
        if langfuse and parent_span:
            try:
                with langfuse.start_as_current_generation(
                    name="llm-query",
                    model=model or self.model,
                    input=msg_list,
                    model_parameters={
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "response_format": str(fmt) if fmt else "text"
                    }
                ) as generation:
                    
                    llm_response = self.client.chat.completions.create(
                        model=model or self.model,
                        messages=msg_list,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format=fmt or {"type": "text"},
                    )
                    
                    response_content = llm_response.choices[0].message.content
                    
                    # Update generation with response and usage
                    usage_dict = None
                    if hasattr(llm_response, 'usage') and llm_response.usage:
                        usage_dict = {
                            "input_tokens": llm_response.usage.prompt_tokens,
                            "output_tokens": llm_response.usage.completion_tokens,
                            "total_tokens": llm_response.usage.total_tokens
                        }
                    
                    generation.update(
                        output=response_content,
                        usage=usage_dict
                    )
                    
                    return response_content
            except Exception as e:
                print(f"Warning: Error with Langfuse generation tracking: {e}")
                # Fallback to regular OpenAI call
                pass
        
        # Fallback or when Langfuse is not available
        llm_response = self.client.chat.completions.create(
            model=model or self.model,
            messages=msg_list,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=fmt or {"type": "text"},
        )
        
        return llm_response.choices[0].message.content

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
            #return valid,f"You can take action {action_str} with parameters {action_params}", action
            return valid, {action_str:action_str,action_params:action_params}, action
       
        except KeyError:
            return False, llm_response, None

    def parse_response(self, llm_response: str, state: Observation.state):
        response_dict = {"action": None, "parameters": None}
        valid = False
        action = None

        try:
            response = json.loads(llm_response)
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
        
        return valid, response_dict, action

    def remove_reasoning(self, text):
        match = re.search(r'</think>(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def check_repetition(self, memory_list):
        repetitions = 0
        past_memories = []
        for memory, goodness in memory_list:
            if memory in past_memories:
                repetitions += 1
            past_memories.append(memory)
        return repetitions

    def get_self_consistent_response(self, messages, temp=0.4, max_tokens=1024, n=3, parent_span=None):
        if parent_span and langfuse:
            try:
                with langfuse.start_as_current_span(name="self-consistency-check") as consistency_span:
                    candidates = []
                    for i in range(n):
                        with langfuse.start_as_current_span(name=f"candidate-{i+1}") as candidate_span:
                            response = self.openai_query(messages, temperature=temp, max_tokens=max_tokens, parent_span=candidate_span)
                            candidates.append(response.strip())

                    counts = Counter(candidates)
                    most_common = counts.most_common(1)
                    
                    consistency_span.update(
                        output={
                            "candidates": candidates,
                            "counts": dict(counts),
                            "selected": most_common[0][0] if most_common else candidates[0]
                        }
                    )
                    
                    if most_common:
                        self.logger.info(f"Self-consistency candidates: {counts}")
                        return most_common[0][0]
                    return candidates[0]
            except Exception as e:
                print(f"Warning: Error with self-consistency tracking: {e}")
        
        # Fallback without Langfuse
        candidates = []
        for i in range(n):
            response = self.openai_query(messages, temperature=temp, max_tokens=max_tokens)
            candidates.append(response.strip())

        counts = Counter(candidates)
        most_common = counts.most_common(1)
        
        if most_common:
            self.logger.info(f"Self-consistency candidates: {counts}")
            return most_common[0][0]
        return candidates[0]

    def get_action_from_obs_react(self, observation: Observation, memory_buf: list) -> tuple:
        # Create main span for this action planning only if Langfuse is available
        if langfuse:
            try:
                # Use context manager directly
                with langfuse.start_as_current_span(
                    name="agent-action-planning",
                    input={
                        "observation_state": observation.state.as_json(),
                        "memory_buffer": [{"action": mem, "goodness": good} for mem, good in memory_buf],
                        "model_config": {
                            "model": self.model,
                            "use_reasoning": self.use_reasoning,
                            "use_reflection": self.use_reflection,
                            "use_self_consistency": self.use_self_consistency
                        }
                    },
                    metadata={
                        "agent_type": "REACT",
                        "memory_length": len(memory_buf),
                        "session_id": self.session_id
                    }
                ) as main_span:
                    # Update trace attributes using langfuse client
                    langfuse.update_current_trace(
                        session_id=self.session_id,
                        user_id="agent",
                        tags=["REACT", "action-planning"]
                    )
                    print(f"✅ Created main span for action planning")
                    return self._execute_planning_logic(observation, memory_buf, main_span)
            except Exception as e:
                print(f"Warning: Could not create Langfuse span: {e}")
                # Fallback without Langfuse
                return self._execute_planning_logic(observation, memory_buf, None)
        else:
            return self._execute_planning_logic(observation, memory_buf, None)

    def _execute_planning_logic(self, observation: Observation, memory_buf: list, main_span=None) -> tuple:
        try:
            self.states.append(observation.state.as_json())
            status_prompt = create_status_from_state(observation.state)
            q1 = self.config['questions'][0]['text']
            q4 = self.config['questions'][3]['text']
            cot_prompt = self.config['prompts']['COT_PROMPT']
            memory_prompt = self.create_mem_prompt(memory_buf)

            repetitions = self.check_repetition(memory_buf)
            
            # Stage 1: Reasoning/Planning
            if langfuse and main_span:
                with langfuse.start_as_current_span(
                    name="stage1-reasoning",
                    input={
                        "repetitions": repetitions,
                        "memory_length": len(memory_buf)
                    }
                ) as stage1_span:
                    messages = [
                        {"role": "user", "content": self.instructions},
                        {"role": "user", "content": status_prompt},
                        {"role": "user", "content": memory_prompt},
                        {"role": "user", "content": q1},
                    ]
                    
                    self.logger.info(f"Text sent to the LLM: {messages}")

                    if self.use_self_consistency:
                        response = self.get_self_consistent_response(messages, temp=repetitions/9, max_tokens=1024, parent_span=stage1_span)
                    else:
                        response = self.openai_query(messages, max_tokens=1024, parent_span=stage1_span)

                    if self.use_reflection:
                        with langfuse.start_as_current_span(name="reflection") as reflection_span:
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
                            response = self.openai_query(reflection_prompt, max_tokens=1024, parent_span=reflection_span)

                    # Optional: parse response if reasoning is expected and outputs <think> ... </think>
                    if self.use_reasoning:
                        response = self.remove_reasoning(response)
                        
                    stage1_span.update(output=response)
                    self.logger.info(f"(Stage 1) Response from LLM: {response}")
            else:
                # Fallback without Langfuse
                messages = [
                    {"role": "user", "content": self.instructions},
                    {"role": "user", "content": status_prompt},
                    {"role": "user", "content": memory_prompt},
                    {"role": "user", "content": q1},
                ]
                
                self.logger.info(f"Text sent to the LLM: {messages}")

                if self.use_self_consistency:
                    response = self.get_self_consistent_response(messages, temp=repetitions/9, max_tokens=1024)
                else:
                    response = self.openai_query(messages, max_tokens=1024)

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
                    response = self.openai_query(reflection_prompt, max_tokens=1024)

                if self.use_reasoning:
                    response = self.remove_reasoning(response)
                    
                self.logger.info(f"(Stage 1) Response from LLM: {response}")

            # Stage 2: Action Selection
            if langfuse and main_span:
                with langfuse.start_as_current_span(name="stage2-action-selection") as stage2_span:
                    messages = [
                        {"role": "user", "content": self.instructions},
                        {"role": "user", "content": status_prompt},
                        {"role": "user", "content": cot_prompt},
                        {"role": "user", "content": response},
                        {"role": "user", "content": memory_prompt},
                        {"role": "user", "content": q4},
                    ]
                    
                    stage2_span.update(input={"messages": messages})
                    self.prompts.append(messages)
                    
                    action_response = self.openai_query(messages, max_tokens=80, fmt={"type": "json_object"}, parent_span=stage2_span)
                    
                    # Validation
                    validated, error_msg = validate_responses.validate_agent_response(action_response)
                    
                    if validated is None:
                        self.logger.info(f"Invalid response format: {action_response} - Error: {error_msg}")
                        
                        try:
                            parsed_response = json.loads(action_response)
                        except json.JSONDecodeError:
                            parsed_response = action_response

                        action_response = json.dumps({
                            "action": "InvalidResponse",
                            "parameters": {
                                "error": error_msg,
                                "original": parsed_response
                            }
                        }, indent=2)

                    if self.use_reasoning:
                        action_response = self.remove_reasoning(action_response)

                    stage2_span.update(output=action_response)
                    
                    self.responses.append(action_response)
                    self.logger.info(f"(Stage 2) Response from LLM: {action_response}")
                    print(f"(Stage 2) Response from LLM: {action_response}")
                    
                    # Parse final response
                    valid, response_dict, action = self.parse_response(action_response, observation.state)
                    
                    # Update main trace with final results using langfuse client
                    langfuse.update_current_trace(
                        output={
                            "valid_response": valid,
                            "selected_action": response_dict,
                            "final_response": action_response
                        }
                    )
                    
                    return valid, response_dict, action
            else:
                # Fallback without Langfuse
                messages = [
                    {"role": "user", "content": self.instructions},
                    {"role": "user", "content": status_prompt},
                    {"role": "user", "content": cot_prompt},
                    {"role": "user", "content": response},
                    {"role": "user", "content": memory_prompt},
                    {"role": "user", "content": q4},
                ]
                
                self.prompts.append(messages)
                
                action_response = self.openai_query(messages, max_tokens=80, fmt={"type": "json_object"})
                
                # Validation
                validated, error_msg = validate_responses.validate_agent_response(action_response)
                
                if validated is None:
                    self.logger.info(f"Invalid response format: {action_response} - Error: {error_msg}")
                    
                    try:
                        parsed_response = json.loads(action_response)
                    except json.JSONDecodeError:
                        parsed_response = action_response

                    action_response = json.dumps({
                        "action": "InvalidResponse",
                        "parameters": {
                            "error": error_msg,
                            "original": parsed_response
                        }
                    }, indent=2)

                if self.use_reasoning:
                    action_response = self.remove_reasoning(action_response)
                
                self.responses.append(action_response)
                self.logger.info(f"(Stage 2) Response from LLM: {action_response}")
                print(f"(Stage 2) Response from LLM: {action_response}")
                
                # Parse final response
                valid, response_dict, action = self.parse_response(action_response, observation.state)
                
                return valid, response_dict, action
                
        except Exception as e:
            self.logger.error(f"Error in _execute_planning_logic: {str(e)}")
            if langfuse and main_span:
                try:
                    langfuse.update_current_trace(
                        output={"error": str(e)},
                        level="ERROR"
                    )
                except Exception as langfuse_error:
                    print(f"Warning: Could not update trace with error: {langfuse_error}")
            raise
    
    def score_action_outcome(self, action_success: bool, reward: float = None, comment: str = None):
        """Score the outcome with correct Langfuse v3 API"""
        if langfuse:
            try:
                score_value = 1.0 if action_success else 0.0
                if reward is not None:
                    score_value = reward
                
                # Get current trace ID
                current_trace_id = langfuse.get_current_trace_id()
                if current_trace_id:
                    langfuse.score(
                        trace_id=current_trace_id,
                        name="action-outcome",
                        value=score_value,
                        comment=comment or ("Success" if action_success else "Failed")
                    )
                    print(f"✅ Scored action outcome: {score_value}")
                else:
                    print("Warning: No active trace to score")
            except Exception as e:
                print(f"Warning: Could not score action outcome: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        pass