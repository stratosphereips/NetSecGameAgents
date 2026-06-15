from netsecgame import Action, ActionType, Data, GameState, Network, Service

from agents.agent_utils import generate_valid_actions_concepts


REMOTE_SERVICE = Service(
    name="ssh",
    type="tcp",
    version="1",
    is_local=False,
)
LOCAL_SERVICE = Service(
    name="local-admin",
    type="tcp",
    version="1",
    is_local=True,
)
SECRET_DATA = Data(owner="user", id="secret")
LOGFILE_DATA = Data(owner="system", id="logfile")


def make_state(
    *,
    controlled_hosts=None,
    known_hosts=None,
    known_services=None,
    known_data=None,
    known_blocks=None,
):
    controlled_hosts = controlled_hosts or {
        "external0",
        "host1",
        "host2",
        "host4",
    }
    known_hosts = known_hosts or controlled_hosts | {"host3"}
    return GameState(
        controlled_hosts=controlled_hosts,
        known_hosts=known_hosts,
        known_services=known_services or {},
        known_data=known_data or {},
        known_networks={Network("net_0_4hosts", 24)},
        known_blocks=known_blocks or {},
    )


def actions_of_type(actions, action_type):
    return [action for action in actions if action.action_type == action_type]


def test_family_filter_ablation_expands_only_selected_family():
    state = make_state(
        known_services={
            "host1": {LOCAL_SERVICE, REMOTE_SERVICE},
            "host3": {LOCAL_SERVICE, REMOTE_SERVICE},
        },
        known_data={
            "external0": {SECRET_DATA},
            "host1": {SECRET_DATA, LOGFILE_DATA},
            "host2": {SECRET_DATA},
        },
    )
    baseline = generate_valid_actions_concepts(state, set())

    switches = {
        "filter_scan_network": ActionType.ScanNetwork,
        "filter_find_services": ActionType.FindServices,
        "filter_exploit_service": ActionType.ExploitService,
        "filter_find_data": ActionType.FindData,
        "filter_exfiltrate_data": ActionType.ExfiltrateData,
    }
    for option, action_type in switches.items():
        ablated = generate_valid_actions_concepts(
            state,
            set(),
            **{option: False},
        )
        baseline_family = set(actions_of_type(baseline, action_type))
        ablated_family = set(actions_of_type(ablated, action_type))
        assert baseline_family
        assert baseline_family < ablated_family
        assert {
            action for action in baseline if action.action_type != action_type
        } == {
            action for action in ablated if action.action_type != action_type
        }


def test_history_ablation_can_apply_globally_or_only_to_network_scans():
    state = make_state()
    baseline = generate_valid_actions_concepts(state, set())
    scan = actions_of_type(baseline, ActionType.ScanNetwork)[0]
    find_services = actions_of_type(baseline, ActionType.FindServices)[0]
    history = {scan, find_services}

    filtered = generate_valid_actions_concepts(state, history)
    scan_repeated = generate_valid_actions_concepts(
        state,
        history,
        allow_repeated_network_scans=True,
    )
    all_repeated = generate_valid_actions_concepts(
        state,
        history,
        allow_repeated_actions=True,
    )

    assert scan not in filtered
    assert find_services not in filtered
    assert scan in scan_repeated
    assert find_services not in scan_repeated
    assert scan in all_repeated
    assert find_services in all_repeated


def test_single_source_uses_one_deterministic_internal_controlled_host():
    state = make_state()
    actions = generate_valid_actions_concepts(state, set(), single_source=True)

    source_based_types = {
        ActionType.ScanNetwork,
        ActionType.FindServices,
        ActionType.ExploitService,
        ActionType.FindData,
    }
    source_hosts = {
        action.parameters["source_host"]
        for action in actions
        if action.action_type in source_based_types
    }
    assert source_hosts == {"host1"}


def test_service_and_exploit_ablation_rules():
    state = make_state(
        known_services={
            "host1": {REMOTE_SERVICE},
            "host3": {LOCAL_SERVICE, REMOTE_SERVICE},
        }
    )

    baseline = generate_valid_actions_concepts(state, set())
    service_rescans = generate_valid_actions_concepts(
        state,
        set(),
        allow_service_rescans=True,
    )
    local_exploits = generate_valid_actions_concepts(
        state,
        set(),
        include_local_services=True,
    )
    controlled_exploits = generate_valid_actions_concepts(
        state,
        set(),
        allow_exploit_controlled_hosts=True,
    )

    assert not any(
        action.parameters["target_host"] == "host1"
        for action in actions_of_type(baseline, ActionType.FindServices)
    )
    assert any(
        action.parameters["target_host"] == "host1"
        for action in actions_of_type(service_rescans, ActionType.FindServices)
    )
    assert not any(
        action.parameters["target_service"] == LOCAL_SERVICE
        for action in actions_of_type(baseline, ActionType.ExploitService)
    )
    assert any(
        action.parameters["target_service"] == LOCAL_SERVICE
        for action in actions_of_type(local_exploits, ActionType.ExploitService)
    )
    assert not any(
        action.parameters["target_host"] == "host1"
        for action in actions_of_type(baseline, ActionType.ExploitService)
    )
    assert any(
        action.parameters["target_host"] == "host1"
        for action in actions_of_type(
            controlled_exploits,
            ActionType.ExploitService,
        )
    )


def test_find_data_ablation_rules():
    state = make_state(known_data={"host1": {SECRET_DATA}})

    baseline = generate_valid_actions_concepts(state, set())
    rescans = generate_valid_actions_concepts(
        state,
        set(),
        allow_find_data_rescans=True,
    )
    no_self_targeting = generate_valid_actions_concepts(
        state,
        set(),
        prohibit_find_data_self_targeting=True,
    )

    assert not any(
        action.parameters["target_host"] == "host1"
        for action in actions_of_type(baseline, ActionType.FindData)
    )
    assert any(
        action.parameters["target_host"] == "host1"
        for action in actions_of_type(rescans, ActionType.FindData)
    )
    assert any(
        action.parameters["target_host"] == action.parameters["source_host"]
        for action in actions_of_type(baseline, ActionType.FindData)
    )
    assert not any(
        action.parameters["target_host"] == action.parameters["source_host"]
        for action in actions_of_type(no_self_targeting, ActionType.FindData)
    )


def test_exfiltration_ablation_rules():
    state = make_state(
        known_data={
            "host1": {SECRET_DATA, LOGFILE_DATA},
            "host2": {SECRET_DATA},
        }
    )

    baseline = generate_valid_actions_concepts(state, set())
    with_logfile = generate_valid_actions_concepts(
        state,
        set(),
        include_logfile_exfiltration=True,
    )
    with_duplicates = generate_valid_actions_concepts(
        state,
        set(),
        allow_duplicate_data_exfiltration=True,
    )
    external_only = generate_valid_actions_concepts(
        state,
        set(),
        exfiltrate_to_external_only=True,
    )

    baseline_exfiltration = actions_of_type(
        baseline,
        ActionType.ExfiltrateData,
    )
    assert not any(
        action.parameters["data"].id == "logfile"
        for action in baseline_exfiltration
    )
    assert any(
        action.parameters["data"].id == "logfile"
        for action in actions_of_type(with_logfile, ActionType.ExfiltrateData)
    )
    assert not any(
        action.parameters["source_host"] == "host1"
        and action.parameters["target_host"] == "host2"
        and action.parameters["data"].id == "secret"
        for action in baseline_exfiltration
    )
    assert any(
        action.parameters["source_host"] == "host1"
        and action.parameters["target_host"] == "host2"
        and action.parameters["data"].id == "secret"
        for action in actions_of_type(
            with_duplicates,
            ActionType.ExfiltrateData,
        )
    )
    assert all(
        "external" in action.parameters["target_host"]
        for action in actions_of_type(external_only, ActionType.ExfiltrateData)
    )


def test_firewall_ablation_restores_pruned_actions():
    state = make_state(
        known_services={"host3": {REMOTE_SERVICE}},
        known_blocks={
            "host1": {"host3"},
            "host3": {"host1"},
            "host2": {"host3"},
            "host3": {"host2"},
            "host4": {"host3"},
            "host3": {"host4"},
        },
    )

    baseline = generate_valid_actions_concepts(state, set())
    firewall_unaware = generate_valid_actions_concepts(
        state,
        set(),
        ignore_firewall=True,
    )

    assert not any(
        action.parameters.get("target_host") == "host3"
        for action in baseline
        if action.action_type
        in {ActionType.FindServices, ActionType.ExploitService}
    )
    assert any(
        action.parameters.get("target_host") == "host3"
        for action in firewall_unaware
        if action.action_type
        in {ActionType.FindServices, ActionType.ExploitService}
    )
