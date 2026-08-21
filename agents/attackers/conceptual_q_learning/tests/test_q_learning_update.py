import logging
import unittest
from unittest.mock import patch

from netsecgame import Action, ActionType, AgentRole, Observation

from agents.attackers.conceptual_q_learning.conceptual_q_agent import QAgent


def _stub_base_agent_init(self, host, port, role):
    self._connection_details = (host, port)
    self._logger = logging.getLogger(self.__class__.__name__)
    self._role = role
    self._socket = None


class TestConceptualQUpdate(unittest.TestCase):
    def make_agent(self, *, alpha=0.1, gamma=0.9):
        with patch(
            "agents.attackers.conceptual_q_learning.conceptual_q_agent.BaseAgent.__init__",
            _stub_base_agent_init,
        ):
            return QAgent(
                "127.0.0.1",
                0,
                role=AgentRole.Attacker,
                alpha=alpha,
                gamma=gamma,
            )

    def test_q_update_uses_incremental_bellman_rule(self):
        agent = self.make_agent(alpha=0.1, gamma=0.9)
        action = Action(
            ActionType.FindData,
            parameters={"source_host": "host1", "target_host": "host2"},
        )
        agent.q_values[(7, action)] = 3.0

        old_q, new_q = agent._update_q_value(
            7,
            action,
            reward=5.0,
            next_q_value=7.0,
            terminal=False,
        )

        self.assertAlmostEqual(old_q, 3.0)
        self.assertAlmostEqual(new_q, 3.83)
        self.assertAlmostEqual(agent.q_values[(7, action)], 3.83)

    def test_terminal_q_update_does_not_bootstrap_from_future_q(self):
        agent = self.make_agent(alpha=0.5, gamma=0.9)
        action = Action(
            ActionType.ExfiltrateData,
            parameters={
                "source_host": "host1",
                "target_host": "external0",
                "data": "secret",
            },
        )
        agent.q_values[(4, action)] = 2.0

        old_q, new_q = agent._update_q_value(
            4,
            action,
            reward=-100.0,
            next_q_value=999.0,
            terminal=True,
        )

        self.assertAlmostEqual(old_q, 2.0)
        self.assertAlmostEqual(new_q, -49.0)
        self.assertAlmostEqual(agent.q_values[(4, action)], -49.0)

    def test_reset_episode_tracking_clears_transient_state(self):
        agent = self.make_agent()
        action = Action(
            ActionType.FindServices,
            parameters={"source_host": "host1", "target_host": "host2"},
        )
        agent.actions_history = {action}
        agent.previous_state = "stale-state"

        agent.reset_episode_tracking()

        self.assertEqual(agent.actions_history, set())
        self.assertIsNone(agent.previous_state)


    def test_testing_action_selection_does_not_mutate_model(self):
        agent = self.make_agent()
        action = Action(
            ActionType.FindServices,
            parameters={"source_host": "host1", "target_host": "host2"},
        )
        observation = Observation(state=object(), reward=0, end=False, info={})

        with (
            patch.object(agent, "generate_valid_actions", return_value=[action]),
            patch(
                "agents.attackers.conceptual_q_learning.conceptual_q_agent.state_as_ordered_string",
                return_value="unseen-state",
            ),
        ):
            selected_action, state_id = agent.select_action(observation, testing=True)

        self.assertEqual(selected_action, action)
        self.assertIsNone(state_id)
        self.assertEqual(agent._str_to_id, {})
        self.assertEqual(agent.q_values, {})


if __name__ == "__main__":
    unittest.main()
