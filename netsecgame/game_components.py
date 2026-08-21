from __future__ import annotations

from pathlib import Path
import enum
import importlib.util
import sys
import types


def _load_game_components_module():
    module_name = "AIDojoCoordinator.game_components"
    if module_name in sys.modules:
        return sys.modules[module_name]

    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "AIDojoCoordinator"
    module_path = package_root / "game_components.py"

    package_name = "AIDojoCoordinator"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_root)]
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AgentRole(str, enum.Enum):
    Attacker = "Attacker"
    Defender = "Defender"
    Benign = "Benign"

    def __repr__(self) -> str:
        return self.value

    def to_string(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AgentRole):
            return self.value == other.value
        if isinstance(other, str):
            return self.value.lower() == other.lower().replace("agentrole.", "")
        return False

    def __hash__(self) -> int:
        return hash(self.value)

    @classmethod
    def from_string(cls, name: str) -> "AgentRole":
        normalized = name.split(".")[-1]
        for role in cls:
            if role.value.lower() == normalized.lower():
                return role
        raise ValueError(f"Invalid AgentRole: {name}")


_game_components = _load_game_components_module()

Action = _game_components.Action
ActionType = _game_components.ActionType
AgentInfo = _game_components.AgentInfo
AgentStatus = _game_components.AgentStatus
Data = _game_components.Data
GameState = _game_components.GameState
GameStatus = _game_components.GameStatus
IP = _game_components.IP
Network = _game_components.Network
Observation = _game_components.Observation
ProtocolConfig = _game_components.ProtocolConfig
Service = _game_components.Service

if not hasattr(_game_components, "AgentRole"):
    _game_components.AgentRole = AgentRole

__all__ = [
    "Action",
    "ActionType",
    "AgentInfo",
    "AgentRole",
    "AgentStatus",
    "Data",
    "GameState",
    "GameStatus",
    "IP",
    "Network",
    "Observation",
    "ProtocolConfig",
    "Service",
]
