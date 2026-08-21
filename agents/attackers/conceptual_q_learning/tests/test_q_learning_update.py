import logging
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from netsecgame import Action, ActionType, AgentRole, AgentStatus, Observation

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


    def test_testing_action_selection_does_not_advance_training_rng(self):
        agent = self.make_agent()
        action = Action(
            ActionType.FindServices,
            parameters={"source_host": "host1", "target_host": "host2"},
        )
        observation = Observation(state=object(), reward=0, end=False, info={})
        training_rng_state = agent._rng.getstate()
        eval_rng_state = agent._eval_rng.getstate()

        with (
            patch.object(agent, "generate_valid_actions", return_value=[action]),
            patch(
                "agents.attackers.conceptual_q_learning.conceptual_q_agent.state_as_ordered_string",
                return_value="unseen-state",
            ),
        ):
            agent.select_action(observation, testing=True)

        self.assertEqual(agent._rng.getstate(), training_rng_state)
        self.assertNotEqual(agent._eval_rng.getstate(), eval_rng_state)


    def test_play_game_returns_sum_of_shaped_step_rewards(self):
        agent = self.make_agent()
        action = Action(
            ActionType.FindServices,
            parameters={"source_host": "host1", "target_host": "host2"},
        )
        initial_observation = Observation(
            state="initial", reward=0, end=False, info={}
        )
        step_observations = [
            Observation(state="state-1", reward=0, end=False, info={}),
            Observation(
                state="state-2",
                reward=0,
                end=True,
                info={"end_reason": AgentStatus.Success},
            ),
        ]

        def as_conceptual(observation):
            return SimpleNamespace(observation=observation, concept_mapping={})

        with (
            patch.object(agent, "generate_valid_actions", return_value=[action]),
            patch.object(agent, "select_action", return_value=(action, None)),
            patch.object(agent, "make_step", side_effect=step_observations),
            patch(
                "agents.attackers.conceptual_q_learning.conceptual_q_agent.convert_concepts_to_actions",
                return_value=action,
            ),
            patch(
                "agents.attackers.conceptual_q_learning.conceptual_q_agent.convert_ips_to_concepts",
                side_effect=as_conceptual,
            ),
        ):
            final_observation, num_steps, episode_return = agent.play_game(
                as_conceptual(initial_observation), episode_num=1, testing=True
            )

        self.assertEqual(final_observation.info["end_reason"], AgentStatus.Success)
        self.assertEqual(num_steps, 2)
        self.assertEqual(episode_return, 999)


if __name__ == "__main__":
    unittest.main()
