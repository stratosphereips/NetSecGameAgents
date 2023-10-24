import sys
from os import path
import re
import logging

import openai
from dotenv import dotenv_values

import jinja2
from tenacity import retry, stop_after_attempt

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from env.game_components import ActionType

config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

local_services = ["can_attack_start_here"]

ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}

COT_PROMPT = """
Example status:
Known networks are 1.1.1.0/24
Known hosts are 2.2.2.2, 1.1.1.2, 2.2.2.3
Controlled hosts are 2.2.2.2, 1.1.1.2
Known data for source host 1.1.1.2: are ('User1', 'SomeData')
Known services for host 1.1.1.1 are "openssh"

Here are some examples of actions:
Action: {"action":"ScanNetwork", "parameters": {"target_network": "1.1.1.0/24"}}
Action: {"action":"ScanServices", "parameters":{"target_host":"2.2.2.3"}}
Action: {"action":"ExploitService", "parameters":{"target_host":"1.1.1.1", "target_service":"openssh"}}
Action: {"action":"FindData", "parameters":{"target_host":"1.1.1.1"}}
Action: {"action":"ExfiltrateData", "parameters":"{'target_host': '2.2.2.2', 'data': ('User1', 'SomeData'), 'source_host': '1.1.1.2'}"}}
End of example.
"""

COT_PROMPT2 = """
Here are some examples of actions:
Action: {"action":"ScanNetwork", "parameters": {"target_network": "1.1.1.0/24"}}
Action: {"action":"ScanServices", "parameters":{"target_host":"2.2.2.3"}}
Action: {"action":"ExploitService", "parameters":{"target_host":"1.1.1.1", "target_service":"openssh"}}
Action: {"action":"FindData", "parameters":{"target_host":"1.1.1.1"}}
Action: {"action":"ExfiltrateData", "parameters": {"target_host": "2.2.2.2", "data": ("User1", "WebData"), "source_host": "1.1.1.2"}}}
End of examples.
"""

EXAMPLE_PROMPT2 = """
Here are some examples of actions:
Action: {"action":"ScanNetwork", "parameters": {"target_network": "1.1.1.0/24"}}
Action: {"action":"ScanServices", "parameters":{"target_host":"2.2.2.3"}}
Action: {"action":"ExploitService", "parameters":{"target_host":"1.1.1.1", "target_service":"openssh"}}
Action: {"action":"FindData", "parameters":{"target_host":"1.1.1.1"}}
Action: {"action":"ExfiltrateData", "parameters": {"target_host": "2.2.2.2", "data": ("User1", "WebData"), "source_host": "1.1.1.2"}}}
End of examples.
"""


INSTRUCTIONS_TEMPLATE = """
You are a pentester and you want to exfiltrate data to target host {{target_host}}.
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
def openai_query(msg_list, max_tokens=60, model="gpt-3.5-turbo"):
    """Send messages to OpenAI API and return the response."""
    llm_response = openai.ChatCompletion.create(
        model=model, messages=msg_list, max_tokens=max_tokens, temperature=0.0
    )
    return llm_response["choices"][0]["message"]["content"]


class LLMAssistant:
    """
    An assistant that takes a state and returns an action to the user.
    """

    def __init__(self, model_name, target_host, memory_len=10) -> None:
        self.model = model_name
        self.target_host = target_host
        self.memory_len = memory_len
        self.memories = []
        self.logger = logging.getLogger("interactive-llm-agent")
        # Create the instructions from the template
        # Once in every instantiation
        jinja_environment = jinja2.Environment()
        template = jinja_environment.from_string(INSTRUCTIONS_TEMPLATE)
        self.instructions = template.render(target_host=self.target_host)

    def create_status_from_state(self, state):
        """Create a status prompt using the current state and the sae memories."""
        contr_hosts = [host.ip for host in state.controlled_hosts]
        known_hosts = [
            str(host) for host in state.known_hosts if host.ip not in contr_hosts
        ]
        known_nets = [str(net) for net in list(state.known_networks)]

        prompt = "Previous actions:\n"
        if len(self.memories) > 0:
            for memory in self.memories:
                prompt += f"You took action {memory[0]} with parameters {memory[1]}\n"
        else:
            prompt += ""

        prompt += "Current status:\n"
        prompt += f"Controlled hosts are {' and '.join(contr_hosts)}\n"
        #    logger.info("Controlled hosts are %s", ' and '.join(contr_hosts))

        prompt += f"Known networks are {' and '.join(known_nets)}\n"
        #    logger.info("Known networks are %s", ' and '.join(known_nets))
        prompt += f"Known hosts are {' and '.join(known_hosts)}\n"
        #    logger.info("Known hosts are %s", ' and '.join(known_hosts))

        if len(state.known_services.keys()) == 0:
            prompt += "Known services are none\n"
        #        logger.info(f"Known services: None")
        for ip_service in state.known_services:
            services = []
            if len(list(state.known_services[ip_service])) > 0:
                for serv in state.known_services[ip_service]:
                    if serv.name not in local_services:
                        services.append(serv.name)
                if len(services) > 0:
                    serv_str = ""
                    for serv in services:
                        serv_str += serv + " and "
                    prompt += f"Known services for host {ip_service} are {serv_str}\n"
                #               logger.info(f"Known services {ip_service, services}")
                else:
                    prompt += "Known services are none\n"
        #                logger.info(f"Known services: None")

        if len(state.known_data.keys()) == 0:
            prompt += "Known data are none\n"
        #        logger.info(f"Known data: None")
        for ip_data in state.known_data:
            if len(state.known_data[ip_data]) > 0:
                host_data = ""
                for known_data in list(state.known_data[ip_data]):
                    host_data += f"({known_data.owner}, {known_data.id}) and "
                prompt += f"Known data for host {ip_data} are {host_data}\n"
        #       logger.info(f"Known data: {ip_data, state.known_data[ip_data]}")
        return prompt

    def create_mem_prompt(self, memory_list):
        """Summarize a list of memories into a few sentences."""
        prompt = ""
        if len(memory_list) > 0:
            for memory in memory_list:
                prompt += f'You have taken action {{"action":"{memory[0]}" with "parameters":"{memory[1]}"}} in the past.\n'
        return prompt

    def parse_response(self, llm_response):
        try:
            regex = r"\{+[^}]+\}\}"
            matches = re.findall(regex, llm_response)
            print("Matches:", matches)
            if len(matches) > 0:
                response = matches[0]
                print("Parsed Response:", response)

                response_ = eval(response)
                action_str = response_["action"]
                action_params = response_["parameters"]
                self.memories.append((action_str, action_params))

                if action_str == "ScanServices":
                    action_str = "FindServices"
                action_str = (
                    f"You can take action {action_str} with parameters {action_params}"
                )
            else:
                action_str = llm_response
            return action_str
        except:
            return llm_response

    def get_action_from_obs(self, observation):
        #  Stage 1
        status_prompt = self.create_status_from_state(observation.state)
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": EXAMPLE_PROMPT2},
            {
                "role": "user",
                "content": "\nSelect a valid action with the correct format and parameters.\nIf an action is in your list of past actions do not chose that action!\nDO NOT REPEAT PAST ACTIONS!",
            },
            {"role": "user", "content": "Action: "},
        ]

        # logger.info(f'Text sent to the LLM: {messages}')

        # messages = [
        #    {"role": "user", "content": self.instructions},
        #    {"role": "user", "content": status_prompt},
        #    {"role": "user", "content": Q1},
        # ]
        response = openai_query(messages, max_tokens=1024, model=self.model)
        self.logger.info(f"(Stage 1) Response from LLM: {response}")
        # memory_prompt = self.create_mem_prompt(self.memories[-self.memory_len :])

        # messages = [
        #    {"role": "user", "content": self.instructions},
        #    {"role": "user", "content": status_prompt},
        #    {"role": "user", "content": COT_PROMPT2},
        #    {"role": "user", "content": response},
        #    {"role": "user", "content": memory_prompt},
        #    {"role": "user", "content": Q4},
        # ]

        # response = openai_query(messages, max_tokens=80, model=self.model)
        # self.logger.info(f"(Stage 2) Response from LLM: {response}")
        action_str = self.parse_response(response)

        return action_str
