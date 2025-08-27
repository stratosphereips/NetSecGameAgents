from __future__ import annotations
import sys
from os import path
import yaml
import logging
import json
import jinja2
import re
from collections import Counter
from typing import Any, List
from tenacity import retry, stop_after_attempt

# Make sure project modules are importable
sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from AIDojoCoordinator.game_components import Observation
from NetSecGameAgents.agents.llm_utils import create_action_from_response

class ConfigLoader:
    """Class to handle loading YAML configurations."""

    @staticmethod
    def load_config(file_name: str = "prompts.yaml") -> dict:
        possible_paths = [
            path.join(path.dirname(__file__), file_name),
            path.join(path.dirname(path.dirname(__file__)), file_name),
            path.join(path.dirname(path.dirname(path.dirname(__file__))), file_name),
        ]
        for yaml_file in possible_paths:
            if path.exists(yaml_file):
                with open(yaml_file, "r") as file:
                    return yaml.safe_load(file)
        raise FileNotFoundError(f"{file_name} not found in expected directories.")

class LLMActionPlannerBase:
    """Base class for LLM-based action planners."""

    def __init__(
        self,
        model_name: str,
        goal: str,
        llm,
        tracer,
        memory_len: int = 10,
        config: dict | None = None,
        use_reasoning: bool = False,
        use_reflection: bool = False,
        use_self_consistency: bool = False,
    ) -> None:
        self.model = model_name
        self.llm = llm
        self.tracer = tracer
        self.config = config or ConfigLoader.load_config()
        self.use_reasoning = use_reasoning
        self.use_reflection = use_reflection
        self.use_self_consistency = use_self_consistency
        self.memory_len = memory_len
        self.logger = logging.getLogger("REACT-agent")
        self.update_instructions(goal.lower())
        self.prompts: List[str] = []
        self.states: List[Any] = []
        self.responses: List[Any] = []
        self.session_id = f"agent-session-{hash(goal)}"

    def get_prompts(self) -> List[str]:
        return self.prompts

    def get_responses(self) -> List[Any]:
        return self.responses

    def get_states(self) -> List[Any]:
        return self.states

    def update_instructions(self, new_goal: str) -> None:
        template = jinja2.Environment().from_string(
            self.config["prompts"]["INSTRUCTIONS_TEMPLATE"]
        )
        self.instructions = template.render(goal=new_goal)

    def create_mem_prompt(self, memory_list: list) -> str:
        prompt = ""
        for memory, goodness in memory_list:
            prompt += (
                f"You have taken action {memory} in the past. This action was {goodness}.\n"
            )
        return prompt

    @retry(stop=stop_after_attempt(3))
    def llm_query(
        self,
        msg_list: list,
        max_tokens: int = 60,
        model: str | None = None,
        fmt=None,
        temperature: float = 0.0,
        parent_span=None,
    ) -> str:
        with self.tracer.start_generation(
            name="llm-query",
            model=model or self.model,
            input=msg_list,
            model_parameters={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": str(fmt) if fmt else "text",
            },
            parent_span=parent_span,
        ) as generation:
            llm_response = self.llm.chat(
                model=model or self.model,
                messages=msg_list,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=fmt or {"type": "text"},
            )
            response_content = llm_response.choices[0].message.content

            if generation and hasattr(llm_response, "usage") and llm_response.usage:
                usage_dict = {
                    "input_tokens": llm_response.usage.prompt_tokens,
                    "output_tokens": llm_response.usage.completion_tokens,
                    "total_tokens": llm_response.usage.total_tokens,
                }
                try:
                    generation.update(output=response_content, usage=usage_dict)
                except Exception:
                    pass
            return response_content

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
            response_dict["parameters"] = llm_response
        except KeyError:
            self.logger.error("Missing keys in LLM response.")
        return valid, response_dict, action

    def remove_reasoning(self, text: str) -> str:
        match = re.search(r"</think>(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def check_repetition(self, memory_list: list) -> int:
        repetitions = 0
        past_memories = []
        for memory, goodness in memory_list:
            if memory in past_memories:
                repetitions += 1
            past_memories.append(memory)
        return repetitions

    def get_self_consistent_response(
        self,
        messages,
        temp: float = 0.4,
        max_tokens: int = 1024,
        n: int = 3,
        parent_span=None,
    ) -> str:
        with self.tracer.start_span(name="self-consistency-check"):
            candidates = []
            for i in range(n):
                with self.tracer.start_span(name=f"candidate-{i+1}"):
                    response = self.llm_query(
                        messages,
                        temperature=temp,
                        max_tokens=max_tokens,
                        parent_span=parent_span,
                    )
                    candidates.append(response.strip())
            counts = Counter(candidates)
            most_common = counts.most_common(1)
            if most_common:
                self.logger.info(f"Self-consistency candidates: {counts}")
                return most_common[0][0]
            return candidates[0]
