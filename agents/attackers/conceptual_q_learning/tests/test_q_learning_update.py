import logging
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

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


    def test_no_actions_training_episode_still_decays_epsilon(self):
        agent = self.make_agent()
        initial_observation = Observation(
            state="initial", reward=0, end=False, info={}
        )
        concept_observation = SimpleNamespace(
            observation=initial_observation, concept_mapping={}
        )

        with patch.object(agent, "generate_valid_actions", return_value=[]):
            final_observation, num_steps, episode_return = agent.play_game(
                concept_observation, episode_num=1000, testing=False
            )

        self.assertEqual(final_observation.info["end_reason"], AgentStatus.Fail)
        self.assertEqual(num_steps, 1)
        self.assertEqual(episode_return, -100)
        self.assertAlmostEqual(agent.current_epsilon, 0.74)
        self.assertEqual(agent.completed_episodes, 1000)


    def test_checkpoint_restores_training_state(self):
        agent = self.make_agent()
        agent.completed_episodes = 1234
        agent.current_epsilon = 0.42
        agent.epsilon_start = 0.75
        agent.epsilon_end = 0.05
        agent.epsilon_max_episodes = 9000
        agent.best_eval_win_rate = 87.5
        agent.best_eval_episode = 1200
        agent.eval_threshold_streak = 2
        agent._rng.random()
        agent._eval_rng.random()
        agent._np_rng.random()

        with TemporaryDirectory() as model_dir:
            agent.store_q_table(model_dir, "checkpoint.pickle")
            expected_training_random = agent._rng.random()
            expected_eval_random = agent._eval_rng.random()
            expected_np_random = agent._np_rng.random()

            restored_agent = self.make_agent()
            with patch(
                "agents.attackers.conceptual_q_learning.conceptual_q_agent._load_legacy_game_components_module"
            ):
                restored_agent.load_q_table(
                    str(Path(model_dir, "checkpoint.pickle"))
                )

        self.assertEqual(restored_agent.completed_episodes, 1234)
        self.assertEqual(restored_agent.current_epsilon, 0.42)
        self.assertEqual(restored_agent.epsilon_start, 0.75)
        self.assertEqual(restored_agent.epsilon_end, 0.05)
        self.assertEqual(restored_agent.epsilon_max_episodes, 9000)
        self.assertEqual(restored_agent.best_eval_win_rate, 87.5)
        self.assertEqual(restored_agent.best_eval_episode, 1200)
        self.assertEqual(restored_agent.eval_threshold_streak, 2)
        self.assertEqual(restored_agent._rng.random(), expected_training_random)
        self.assertEqual(restored_agent._eval_rng.random(), expected_eval_random)
        self.assertEqual(restored_agent._np_rng.random(), expected_np_random)


    def test_eval_threshold_streak_requires_consecutive_hits(self):
        agent = self.make_agent()

        self.assertEqual(agent.update_eval_threshold_streak(95.0, 95.0), 1)
        self.assertEqual(agent.update_eval_threshold_streak(97.0, 95.0), 2)
        self.assertEqual(agent.update_eval_threshold_streak(94.9, 95.0), 0)
        self.assertEqual(agent.update_eval_threshold_streak(96.0, 95.0), 1)


if __name__ == "__main__":
    unittest.main()
