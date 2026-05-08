from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from netsecgame import Observation


def filter_log_files_from_state(observation: Observation) -> Observation:
    """Removes verbose 'logfile' entities from the observation state."""
    for host in list(observation.state.known_data.keys()):
        data_list = observation.state.known_data[host]
        filtered_data = [data for data in data_list if data.id != "logfile"]
        if len(filtered_data) > 0:
            observation.state.known_data[host] = filtered_data
        else:
            del observation.state.known_data[host]
    return observation


@runtime_checkable
class HasEpisodeStats(Protocol):
    input_tokens: int
    output_tokens: int
    won: bool
    steps: int
    total_reward: float


@dataclass
class RunStats:
    """Accumulates per-episode token, step, reward, and outcome statistics across a run."""

    _wins: int = field(default=0, init=False)
    _episodes: int = field(default=0, init=False)
    _input_tokens: list = field(default_factory=list, init=False)
    _output_tokens: list = field(default_factory=list, init=False)
    _steps: list = field(default_factory=list, init=False)
    _rewards: list = field(default_factory=list, init=False)

    def record(
        self,
        won: bool,
        input_tokens: int,
        output_tokens: int,
        steps: int = 0,
        total_reward: float = 0.0,
    ) -> None:
        self._episodes += 1
        if won:
            self._wins += 1
        self._input_tokens.append(input_tokens)
        self._output_tokens.append(output_tokens)
        self._steps.append(steps)
        self._rewards.append(total_reward)

    def record_outcome(self, outcome: "HasEpisodeStats") -> None:
        self.record(
            won=outcome.won,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            steps=getattr(outcome, "steps", 0),
            total_reward=getattr(outcome, "total_reward", 0.0),
        )

    @property
    def episodes(self) -> int:
        return self._episodes

    @property
    def wins(self) -> int:
        return self._wins

    def _stats(self, values: list) -> dict:
        if not values:
            return {"total": 0, "mean": 0.0, "min": 0, "max": 0}
        return {
            "total": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def summary(self) -> str:
        n = self._episodes
        if n == 0:
            return "RunStats: no episodes recorded."
        win_rate = self._wins / n
        total_tok = [i + o for i, o in zip(self._input_tokens, self._output_tokens)]
        in_s = self._stats(self._input_tokens)
        out_s = self._stats(self._output_tokens)
        tot_s = self._stats(total_tok)
        steps_s = self._stats(self._steps)
        rew_s = self._stats(self._rewards)
        lines = [
            f"=== Run Summary ({n} episodes) ===",
            f"  win rate : {self._wins}/{n} ({win_rate:.1%})",
            f"  steps           — mean: {steps_s['mean']:>8,.1f}  min: {steps_s['min']:>8}  max: {steps_s['max']:>8}  total: {steps_s['total']:>9,}",
            f"  reward          — mean: {rew_s['mean']:>8,.2f}  min: {rew_s['min']:>8.2f}  max: {rew_s['max']:>8.2f}  total: {rew_s['total']:>9,.2f}",
            f"  tokens (input)  — total: {in_s['total']:>9,}  mean: {in_s['mean']:>8,.0f}  min: {in_s['min']:>8,}  max: {in_s['max']:>8,}",
            f"  tokens (output) — total: {out_s['total']:>9,}  mean: {out_s['mean']:>8,.0f}  min: {out_s['min']:>8,}  max: {out_s['max']:>8,}",
            f"  tokens (total)  — total: {tot_s['total']:>9,}  mean: {tot_s['mean']:>8,.0f}  min: {tot_s['min']:>8,}  max: {tot_s['max']:>8,}",
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())
