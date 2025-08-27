"""
@file llm_action_planner.py

@brief Implementation of an LLM-based action planner for reactive agent systems.

This module defines the ``LLMActionPlanner`` which orchestrates interactions with a
language model to plan actions via the ReAct technique. It leverages the generic
``LLMActionPlannerBase`` for provider-agnostic LLM access and optional tracing
support. The core logic mirrors the original implementation but delegates LLM and
tracer specifics to injected dependencies.

Most of the code is adapted from the original ``assistant.py`` from the
``interactive_tui`` agent.

@author Maria Rigaki - maria.rigaki@aic.fel.cvut.cz
@author Harpo Maxx - harpomaxx@gmail.com

@date [Date]
"""

from __future__ import annotations
import json
import logging
from AIDojoCoordinator.game_components import ActionType, Observation
from NetSecGameAgents.agents.llm_utils import create_status_from_state
from .llm_action_planner_base import LLMActionPlannerBase
import validate_responses


ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}

class LLMActionPlanner(LLMActionPlannerBase):
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
        super().__init__(
            model_name=model_name,
            goal=goal,
            llm=llm,
            tracer=tracer,
            memory_len=memory_len,
            config=config,
            use_reasoning=use_reasoning,
            use_reflection=use_reflection,
            use_self_consistency=use_self_consistency,
        )

    def get_action_from_obs_react(self, observation: Observation, memory_buf: list) -> tuple:
        with self.tracer.start_span(
            name="agent-action-planning",
            input={
                "observation_state": observation.state.as_json(),
                "memory_buffer": [{"action": mem, "goodness": good} for mem, good in memory_buf],
                "model_config": {
                    "model": self.model,
                    "use_reasoning": self.use_reasoning,
                    "use_reflection": self.use_reflection,
                    "use_self_consistency": self.use_self_consistency,
                },
            },
            metadata={"agent_type": "REACT", "memory_length": len(memory_buf), "session_id": self.session_id},
        ) as main_span:
            self.states.append(observation.state.as_json())
            status_prompt = create_status_from_state(observation.state)
            q1 = self.config["questions"][0]["text"]
            q4 = self.config["questions"][3]["text"]
            cot_prompt = self.config["prompts"]["COT_PROMPT"]
            memory_prompt = self.create_mem_prompt(memory_buf)
            repetitions = self.check_repetition(memory_buf)

            with self.tracer.start_span(
                name="stage1-reasoning",
                input={"repetitions": repetitions, "memory_length": len(memory_buf)},
                parent_span=main_span,
            ) as stage1_span:
                messages = [
                    {"role": "user", "content": self.instructions},
                    {"role": "user", "content": status_prompt},
                    {"role": "user", "content": memory_prompt},
                    {"role": "user", "content": q1},
                ]
                if self.use_self_consistency:
                    response = self.get_self_consistent_response(messages, temp=repetitions/9, max_tokens=1024, parent_span=stage1_span)
                else:
                    response = self.llm_query(messages, max_tokens=1024, parent_span=stage1_span)
                if self.use_reflection:
                    with self.tracer.start_span(name="reflection", parent_span=stage1_span) as reflection_span:
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
                                """,
                            }
                        ]
                        response = self.llm_query(reflection_prompt, max_tokens=1024, parent_span=reflection_span)
                if self.use_reasoning:
                    response = self.remove_reasoning(response)
                try:
                    stage1_span.update(output=response)
                except Exception:
                    pass

            with self.tracer.start_span(name="stage2-action-selection", parent_span=main_span) as stage2_span:
                messages = [
                    {"role": "user", "content": self.instructions},
                    {"role": "user", "content": status_prompt},
                    {"role": "user", "content": cot_prompt},
                    {"role": "user", "content": response},
                    {"role": "user", "content": memory_prompt},
                    {"role": "user", "content": q4},
                ]
                try:
                    stage2_span.update(input={"messages": messages})
                except Exception:
                    pass
                self.prompts.append(messages)
                action_response = self.llm_query(messages, max_tokens=80, fmt={"type": "json_object"}, parent_span=stage2_span)
                validated, error_msg = validate_responses.validate_agent_response(action_response)
                if validated is None:
                    try:
                        parsed_response = json.loads(action_response)
                    except json.JSONDecodeError:
                        parsed_response = action_response
                    action_response = json.dumps({"action": "InvalidResponse", "parameters": {"error": error_msg, "original": parsed_response}}, indent=2)
                if self.use_reasoning:
                    action_response = self.remove_reasoning(action_response)
                try:
                    stage2_span.update(output=action_response)
                except Exception:
                    pass
                self.responses.append(action_response)
                valid, response_dict, action = self.parse_response(action_response, observation.state)
                try:
                    stage2_span.update(metadata={"parsed": response_dict})
                except Exception:
                    pass
                return valid, response_dict, action

    def score_action_outcome(self, action_success: bool, reward: float = None, comment: str = None) -> None:
        with self.tracer.start_span(
            name="action-outcome",
            input={"success": action_success, "reward": reward, "comment": comment},
        ):
            pass
