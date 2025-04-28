import json
import logging

# from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import dotenv_values

import jinja2
from tenacity import retry, stop_after_attempt


from AIDojoCoordinator.game_components import ActionType, Observation
from NetSecGameAgents.agents.llm_utils import (
    create_action_from_response,
    create_status_from_state,
)


local_services = ["can_attack_start_here"]

ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}

COT_PROMPT = """
Here are some examples of actions:
Action: {"action":"ScanNetwork", "parameters": {"target_network": "1.1.1.0/24", 'source_host': '2.2.2.2'}}
Action: {"action":"ScanServices", "parameters":{"target_host":"2.2.2.3", 'source_host': '2.2.2.2'}}
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


@retry(stop=stop_after_attempt(3))
async def openai_query(
    client: AsyncOpenAI,
    msg_list: list,
    max_tokens: int = 60,
    model: str = "gpt-3.5-turbo",
    fmt={"type": "text"},
):
    """Send messages to OpenAI API and return the response."""
    llm_response = await client.chat.completions.create(
        model=model,
        messages=msg_list,
        max_tokens=max_tokens,
        temperature=0.0,
        response_format=fmt,
    )
    return llm_response.choices[0].message.content


class LLMAssistant:
    """
    An assistant that takes a state and returns an action to the user.
    """

    def __init__(self, model_name: str, goal: str, memory_len: int = 10, api_url=None):
        self.model = model_name

        if "gpt" in self.model:
            config = dotenv_values(".env")
            self.client = AsyncOpenAI(api_key=config["OPENAI_API_KEY"])
        else:
            self.client = AsyncOpenAI(base_url=api_url, api_key="ollama")
        self.memory_len = memory_len
        # self.memories = []
        self.logger = logging.getLogger("Interactive-TUI-agent")
        # Create the instructions from the template
        # Once in every instantiation
        self.update_instructions(goal.lower())

    def update_instructions(self, new_goal: str) -> None:
        jinja_environment = jinja2.Environment()
        template = jinja_environment.from_string(INSTRUCTIONS_TEMPLATE)
        self.instructions = template.render(goal=new_goal)

    def create_mem_prompt(self, memory_list: list) -> str:
        """Summarize a list of memories into a few sentences."""
        prompt = ""
        if len(memory_list) > 0:
            for memory, goodness in memory_list:
                if goodness:
                    prompt += f"You have taken action {str(memory)} in the past. This action was helpful.\n"
                else:
                    prompt += f"You have taken action {str(memory)} in the past. This action was not helpful.\n"
        return prompt

    def parse_response(self, llm_response: str, state: Observation.state):
        try:
            response = json.loads(llm_response)
        except:
            self.logger(f"JSON excpetion {type(llm_response)}")
            return llm_response, None

        try:
            action_str = response["action"]
            action_params = response["parameters"]
            # self.memories.append((action_str, action_params))

            _, action = create_action_from_response(response, state)
            # if action_str == "ScanServices":
            #     action_str = "FindServices"
            action_output = (
                f"You can take action {action_str} with parameters {action_params}"
            )
            return action_output, action
        except:
            return llm_response, None

    async def get_action_from_obs_react(
        self, observation: Observation, memory_buf: list
    ) -> tuple:
        """
        Use the ReAct architecture for the assistant
        """
        #  Stage 1
        status_prompt = create_status_from_state(observation.state)

        messages = [
            {"role": "user", "content": self.instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": Q1},
        ]
        self.logger.info(f"Text sent to the LLM: {messages}")

        response = await openai_query(
            self.client, messages, max_tokens=1024, model=self.model
        )
        self.logger.info(f"(Stage 1) Response from LLM: {response}")

        # Stage 2
        memory_prompt = self.create_mem_prompt(memory_buf)

        messages = [
            {"role": "user", "content": self.instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": COT_PROMPT},
            {"role": "user", "content": response},
            {"role": "user", "content": memory_prompt},
            {"role": "user", "content": Q4},
        ]

        response = await openai_query(
            self.client,
            messages,
            max_tokens=80,
            model=self.model,
            fmt={"type": "json_object"},
        )
        self.logger.info(f"(Stage 2) Response from LLM: {response}")
        action_str, action = self.parse_response(response, observation.state)

        return action_str, action
