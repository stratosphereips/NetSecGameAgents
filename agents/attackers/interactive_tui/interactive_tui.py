#
# Author:  Maria Rigaki - maria.rigaki@aic.fel.cvut.cz
import os
import logging
import ipaddress
import argparse
import asyncio
from textual.app import App, ComposeResult, Widget
from textual.widgets import Tree, Button, RichLog, Select, Input
from textual.containers import Vertical, VerticalScroll, Horizontal
from textual.validation import Function
from textual import on
from textual.reactive import reactive
from NetSecGameAgents.agents.attackers.interactive_tui.assistant import LLMAssistant
from AIDojoCoordinator.game_components import Network, IP, ActionType, Action, GameState, Observation, AgentStatus
from NetSecGameAgents.agents.base_agent import BaseAgent
log_filename = os.path.dirname(os.path.abspath(__file__)) + "/interactive_tui_agent.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger("Interactive-TUI-agent")
logger.info("Start")


def is_valid_ip(ip_addr: str) -> bool:
    """Validate if the input string is an IPv4 or IPv6 address"""
    try:
        ipaddress.ip_address(ip_addr)
        return True
    except ValueError:
        return False


def is_valid_net(net_addr: str) -> bool:
    """Validate if the input string is an IPv4 or IPv6 network"""

    # Do not accept empty mask
    if "/" not in net_addr:
        return False
    try:
        ipaddress.ip_network(net_addr, strict=True)
        return True
    except (ipaddress.NetmaskValueError, ipaddress.AddressValueError, ValueError):
        return False


class TreeState(Widget):
    tree = reactive(Tree("State", classes="box"))

    def __init__(self, obs: Observation):
        super().__init__()
        self.init_obs = obs
        # self.tree =

    def _create_tree_from_obs(self, observation: Observation) -> Tree:
        # tree: Tree[dict] = Tree("State", classes="box")
        # tree = self.tree
        self.tree.root.expand()
        state = observation.state

        contr_host_list = [host for host in state.controlled_hosts]
        known_host_list = [
            host for host in state.known_hosts if host not in contr_host_list
        ]

        networks = self.tree.root.add("Known Networks", expand=True)
        for network in state.known_networks:
            networks.add(str(network))

        known_hosts = self.tree.root.add("Known Hosts", expand=True)
        for host in known_host_list:
            h = known_hosts.add(str(host))
            for s_host in state.known_services:
                if s_host == host:
                    node = h.add("Services", expand=True)
                    for service in state.known_services[s_host]:
                        node.add_leaf(service.name)

        owned_hosts = self.tree.root.add("Controlled Hosts", expand=True)
        for host in contr_host_list:
            h = owned_hosts.add(str(host))

            # Add any services
            for s_host in state.known_services:
                if s_host == host:
                    node = h.add("Services", expand=True)
                    for service in state.known_services[s_host]:
                        node.add_leaf(service.name)
            # Add known data
            for d_host in state.known_data:
                if d_host == host:
                    node = h.add("Data", expand=True)
                    for datum in state.known_data[host]:
                        node.add_leaf(f"{datum.owner} - {datum.id}")

        # return tree

    def compose(self) -> ComposeResult:
        self._create_tree_from_obs(self.init_obs)
        yield self.tree


class InteractiveTUI(App):
    """App to display key events."""

    CSS_PATH = "layout.tcss"

    def __init__(
        self,
        host: str,
        port: int,
        role: str,
        mode: str,
        llm: str,
        api_url: str,
        memory_len: int,
        max_repetitions: int,
    ):
        super().__init__()
        self.returns = 0
        self.next_action = None
        self.src_host_input = ""
        self.target_host_input = ""
        self.network_input = ""
        self.service_input = ""
        self.data_input = ""
        self.agent = BaseAgent(host, port, role)
        self.agent.register()
        self.current_obs = self.agent.request_game_reset()
        self.mode = mode

        # Keep track of the actions played previously
        # and how many to send to the assistant for the prompt creation
        self.memory_len = memory_len
        self.memory_buf = []
        self.repetitions = 1
        self.stop = False
        self.max_repetitions = max_repetitions

        if llm != "None":
            self.model = llm
        else:
            self.model = None

        if self.model is not None:
            self.assistant = LLMAssistant(
                self.model,
                self.current_obs.info["goal_description"],
                memory_len,
                api_url,
            )

    def compose(self) -> ComposeResult:
        """
        Creates the layout
        """
        yield Vertical(TreeState(obs=self.current_obs), classes="box", id="tree")
        yield Select(
            [
                ("ScanNetwork", ActionType.ScanNetwork),
                ("ScanServices", ActionType.FindServices),
                ("ExploitService", ActionType.ExploitService),
                ("FindData", ActionType.FindData),
                ("ExfiltrateData", ActionType.ExfiltrateData),
            ],
            prompt="Select Action",
            classes="box",
        )
        if self.mode == "guided":
            yield Vertical(
                VerticalScroll(Select([], prompt="Select source host", id="src_host")),
                VerticalScroll(Select([], prompt="Select network", id="network")),
                VerticalScroll(
                    Select(
                        [],
                        prompt="Select target host",
                        id="target_host",
                    )
                ),
                VerticalScroll(Select([], prompt="Select service", id="service")),
                VerticalScroll(Select([], prompt="Select data", id="data")),
                classes="params",
            )
        else:
            yield Vertical(
                VerticalScroll(
                    Input(
                        placeholder="Source Host",
                        id="src_host",
                        validators=[
                            Function(is_valid_ip, "This is not a valid IP address."),
                        ],
                    ),
                    Input(
                        placeholder="Network",
                        id="network",
                        validators=[
                            Function(is_valid_net, "This is not a valid Network.")
                        ],
                    ),
                    Input(
                        placeholder="Target Host",
                        id="target_host",
                        validators=[Function(is_valid_ip, "This is not a valid IP.")],
                    ),
                    Input(placeholder="Service", id="service"),
                    Input(placeholder="Data", id="data"),
                    classes="box",
                )
            )
        yield RichLog(classes="box", id="textarea", highlight=True, markup=True)
        yield Horizontal(
            Button("Take Action", variant="primary", id="act"),
            Button("Assist", variant="primary", id="assist"),
            Button("Assist & Play", variant="warning", id="play"),
        )
        yield Button("Hack the Future", variant="error", id="hack")

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        """Handles the selections of the drop down menus"""
        match event._sender.id:
            case "src_host":
                self.src_host_input = event.value
                return
            case "network":
                self.network_input = event.value
                return
            case "target_host":
                self.target_host_input = event.value
                return
            case "service":
                self.service_input = event.value
                return
            case "data":
                self.data_input = event.value
                return
            case _:
                # otherwise it is the action selector
                self.next_action = event.value
                # log = self.query_one("RichLog")
                # log.write(f"Action selected {self.next_action}, {event.value}")

        state = self.current_obs.state

        if self.mode == "guided":
            net_input = self.query_one("#network", Select)
            target_input = self.query_one("#target_host", Select)
            service_input = self.query_one("#service", Select)
            data_input = self.query_one("#data", Select)

            contr_hosts = [(str(host), str(host)) for host in state.controlled_hosts]
            src_input = self.query_one("#src_host", Select)
            src_input.set_options(contr_hosts)

            known_hosts = [
                (str(host), str(host))
                for host in state.known_hosts
                if host not in state.controlled_hosts
            ]

            match event.value:
                case ActionType.ScanNetwork:
                    networks = [
                        (str(net), str(net))
                        for net in self.current_obs.state.known_networks
                    ]
                    net_input.set_options(networks)
                case ActionType.FindServices:
                    target_input.set_options(known_hosts)
                case ActionType.ExploitService:
                    target_input.set_options(known_hosts)

                    services = set()
                    for host in state.known_services:
                        if host not in state.controlled_hosts:
                            for serv in state.known_services[host]:
                                services.add((serv.name, serv.name))
                    service_input.set_options(services)

                case ActionType.FindData:
                    target_input.set_options(contr_hosts)

                case ActionType.ExfiltrateData:
                    target_input.set_options(contr_hosts)

                    data = set()
                    for host in state.known_data:
                        for d in state.known_data[host]:
                            data.add((d.id, d.id))
                    data_input.set_options(data)

        else:
            net_input = self.query_one("#network", Input)
            target_input = self.query_one("#target_host", Input)
            service_input = self.query_one("#service", Input)
            data_input = self.query_one("#data", Input)

        # Set proper visibility
        if event.value == ActionType.ScanNetwork:
            net_input.visible = True
            service_input.visible = False
            target_input.visible = False
            data_input.visible = False
        elif event.value == ActionType.FindServices:
            net_input.visible = False
            service_input.visible = False
            target_input.visible = True
            data_input.visible = False
        elif event.value == ActionType.ExploitService:
            net_input.visible = False
            service_input.visible = True
            target_input.visible = True
            data_input.visible = False
        elif event.value == ActionType.FindData:
            net_input.visible = False
            service_input.visible = False
            target_input.visible = True
            data_input.visible = False
        elif event.value == ActionType.ExfiltrateData:
            net_input.visible = False
            service_input.visible = False
            target_input.visible = True
            data_input.visible = True
        else:
            net_input.visible = True
            service_input.visible = True
            target_input.visible = True
            data_input.visible = True

    @on(Input.Changed)
    def handle_inputs(self, event: Input.Changed) -> None:
        """
        Handles the manual inputs that are types by the user.
        """
        # log = self.query_one("RichLog")
        # log.write(f"Input received: {event.value}")
        if event._sender.id == "src_host":
            if event.validation_result.is_valid:
                self.src_host_input = event.value
            else:
                self.src_host_input = ""
        elif event._sender.id == "network":
            if event.validation_result.is_valid:
                self.network_input = event.value
            else:
                self.network_input = ""
        elif event._sender.id == "target_host":
            if event.validation_result.is_valid:
                self.target_host_input = event.value
            else:
                self.src_host_input = ""
        elif event._sender.id == "service":
            self.service_input = event.value
        elif event._sender.id == "data":
            self.data_input = event.value

    @on(Button.Pressed)
    async def do_something(self, event: Button.Pressed) -> None:
        """
        Handles the button events.
        """
        log = self.query_one("RichLog")
        if event.button.id == "act":
            action = self.generate_action(self.current_obs.state)

            if action is not None:
                self.update_state(action)

                # Take the first node of TreeState which contains the tree
                tree_state = self.query_one(TreeState)
                tree = tree_state.children[0]
                self.update_tree(tree)

        elif event.button.id == "assist":
            if self.model is not None:
                log.write("Waiting for the LLM...")

                async def do_ask_llm():
                    (
                        act_str,
                        action,
                    ) = await self.assistant.get_action_from_obs_react(
                        self.current_obs, self.memory_buf[-self.memory_len :]
                    )

                    if action is not None:
                        if action.type.name == "FindServices":
                            action_name = "ScanServices"
                        else:
                            action_name = action.type.name

                        msg = f"[bold yellow]:robot: Assistant proposed:[/bold yellow] {action_name} with {action.parameters}"
                        log.write(msg)
                    else:
                        msg = f"[bold red]:robot: Assistant proposed (invalid):[/bold red] {act_str}"
                        log.write(msg)

                asyncio.create_task(do_ask_llm())
        else:
            if self.model is not None:
                log.write(":hourglass: Waiting for the LLM...")

                async def do_ask_llm():
                    act_str, action = await self.assistant.get_action_from_obs_react(
                        self.current_obs, self.memory_buf[-self.memory_len :]
                    )

                    if action is not None:
                        # To remove the discrepancy between scan and find services
                        if action.type.name == "FindServices":
                            action_name = "ScanServices"
                        else:
                            action_name = action.type.name
                        msg = f"[bold yellow]:robot: Assistant played:[/bold yellow] {action_name} with {action.parameters}"
                        log.write(msg)
                        log.write(":hourglass: LLM finished.")
                        # if event.button.id == "hack":
                        self.update_state(action)

                        tree_state = self.query_one(TreeState)
                        tree = tree_state.children[0]
                        self.update_tree(tree)
                    else:
                        msg = f"[bold red]:robot: Assistant proposed (invalid):[/bold red] {act_str}"
                        log.write(msg)
                        log.write(":hourglass: LLM finished.")
                    # self.notify(
                    #     message=msg, title="LLM Action", timeout=15, severity="warning"
                    # )

                    # Redo if hack the planet
                    if event.button.id == "hack":
                        if self.repetitions < self.max_repetitions:
                            self.repetitions += 1
                            self.post_message(Button.Pressed(Button(id="hack")))
                            return
                        else:
                            self.repetitions = 1

                asyncio.create_task(do_ask_llm())
            else:
                log.write(
                    "[bold red]No assistant is available at the moment.[/bold red]"
                )

    def update_state(self, action: Action) -> None:
        """
        Take an action and receive the new state from the environment.
        """
        # Get next observation of the environment
        log = self.query_one("RichLog")
        log.write(":gear: Taking an action in the environment.")
        next_observation = self.agent.make_step(action)
        if next_observation.state != self.current_obs.state:
            good_action = True
        else:
            good_action = False

        # Collect reward
        self.returns += next_observation.reward
        # Move to next state
        self.current_obs = next_observation
        self.memory_buf.append((action, good_action))

        if next_observation.end:
            log = self.query_one("RichLog")
            if next_observation.info["end_reason"] == AgentStatus.Success:
                log.write(
                    f"[bold green]:tada: :fireworks: :trophy: You won! Total return: {self.returns}[/bold green]",
                )
                self.notify(f"You won! Total return: {self.returns}", timeout=20)
            else:
                log.write(
                    f"[bold red]:x: :sob: You lost! Total return: {self.returns}[/bold red]"
                )
                self.notify(
                    f"You lost! Total return: {self.returns}",
                    severity="error",
                    timeout=10,
                )
            self._clear_state()

    def update_tree(self, tree: Widget) -> None:
        """Update the tree with the new state"""

        # Get the new state
        new_state = self.current_obs.state

        # First remove all the children and then rebuild the tree
        # This is faster than looking at the delta
        tree.root.remove_children()

        contr_host_list = new_state.controlled_hosts
        known_host_list = [
            host for host in new_state.known_hosts if host not in contr_host_list
        ]

        networks = tree.root.add("Known Networks", expand=True)
        for network in new_state.known_networks:
            networks.add(str(network))

        known_hosts = tree.root.add("Known Hosts", expand=True)
        for host in known_host_list:
            h = known_hosts.add(str(host), expand=True)
            for s_host in new_state.known_services:
                if s_host == host:
                    node = h.add("Services", expand=True)
                    for service in new_state.known_services[s_host]:
                        node.add_leaf(service.name)

        owned_hosts = tree.root.add("Controlled Hosts", expand=True)
        for host in contr_host_list:
            h = owned_hosts.add(str(host), expand=True)

            # Add any services
            for s_host in new_state.known_services:
                if s_host == host:
                    node = h.add("Services", expand=True)
                    for service in new_state.known_services[s_host]:
                        node.add_leaf(service.name)
            # Add known data
            for d_host in new_state.known_data:
                if d_host == host:
                    node = h.add("Data", expand=True)
                    for datum in new_state.known_data[host]:
                        node.add_leaf(f"{datum.owner} - {datum.id}")

    def generate_action(self, state: GameState) -> Action:
        """Generate a valid action from the user inputs"""
        action = None
        log = self.query_one("RichLog")
        # log.write(self.next_action)
        if self.next_action == ActionType.ScanNetwork:
            if self.src_host_input != "" and self.network_input != "":
                parameters = {
                    "source_host": IP(self.src_host_input),
                    "target_network": Network(
                        self.network_input[:-3], mask=int(self.network_input[-2:])
                    ),
                }
                action = Action(action_type=self.next_action, parameters=parameters)
            else:
                self.notify("Please provide valid inputs", severity="error")
        elif self.next_action in [ActionType.FindServices, ActionType.FindData]:
            if self.src_host_input != "" and self.target_host_input != "":
                parameters = {
                    "source_host": IP(self.src_host_input),
                    "target_host": IP(self.target_host_input),
                }
                action = Action(action_type=self.next_action, parameters=parameters)
            else:
                self.notify("Please provide valid inputs", severity="error")
        elif self.next_action == ActionType.ExploitService:
            if self.src_host_input != "" and self.target_host_input != "":
                for host, service_list in state.known_services.items():
                    if IP(self.target_host_input) == host:
                        for service in service_list:
                            if self.service_input == service.name:
                                parameters = {
                                    "source_host": IP(self.src_host_input),
                                    "target_host": IP(self.target_host_input),
                                    "target_service": service,
                                }
                                action = Action(
                                    action_type=self.next_action, parameters=parameters
                                )
                                break
            else:
                self.notify("Please provide valid inputs", severity="error")
        elif self.next_action == ActionType.ExfiltrateData:
            if self.src_host_input != "" and self.target_host_input != "":
                for host, data_items in state.known_data.items():
                    # log.write(f"{str(state.known_data.items())}")
                    if IP(self.src_host_input) == host:
                        for datum in data_items:
                            if self.data_input == datum.id:
                                parameters = {
                                    "source_host": IP(self.src_host_input),
                                    "target_host": IP(self.target_host_input),
                                    "data": datum,
                                }
                                action = Action(
                                    action_type=self.next_action, parameters=parameters
                                )
                            else:
                                parameters = self.data_input
            else:
                self.notify("Please provide valid inputs", severity="error")
        else:
            log.write(
                f"[bold red]Invalid input: {self.next_action} with {parameters}[/bold red]"
            )
            logger.info(
                f"Invalid input from user: {self.next_action} with {parameters}"
            )

        if action is None:
            log.write(
                f"[bold red]Please select a valid action and parameters[/bold red]"
            )
            # logger.info(f"Random action due to error: {str(action)}")

        else:
            if action.type.name == "FindServices":
                action_name = "ScanServices"
            else:
                action_name = action.type.name
            log.write(
                f"[bold blue]:woman: Action selected:[/bold blue] {action_name} with {action.parameters}"
            )
            logger.info(f"User selected action: {str(action)}")

        return action

    def _clear_state(self) -> None:
        """Reset the state and variables"""
        logger.info("Reset the environment and state")
        self.current_obs = self.agent.request_game_reset()
        if self.model is not None:
            self.assistant.update_instructions(
                self.current_obs.info["goal_description"]
            )
        self.next_action = None
        self.src_host_input = ""
        self.target_host_input = ""
        self.network_input = ""
        self.service_input = ""
        self.data_input = ""
        self.memory_buf = []
        self.repetitions = self.max_repetitions

        selectors = self.query(Select)
        for selector in selectors:
            selector.clear()

        for inp in self.query(Input):
            inp.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--role", help="Role of the agent", default="Attacker", choices=["Attacker"]
    )
    parser.add_argument(
        "--mode", type=str, choices=["guided", "normal"], default="guided"
    )
    parser.add_argument(
        "--llm",
        choices=[
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o-mini",
            "zephyr",
            "llama2",
            "None",
        ],
        default="None",
    )
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:11434/v1/")
    parser.add_argument("--memory_len", type=int, default=10)
    parser.add_argument("--max_repetitions", type=int, default=10)
    args = parser.parse_args()

    logger.info("Creating the agent")
    app = InteractiveTUI(
        args.host,
        args.port,
        args.role,
        args.mode,
        args.llm,
        args.api_url,
        args.memory_len,
        args.max_repetitions,
    )
    app.run()