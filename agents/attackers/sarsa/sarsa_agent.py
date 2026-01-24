# Authors:  Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
from os import path, makedirs
import numpy as np
import random
import pickle
import argparse
import logging

from netsecgame import Action, GameState, BaseAgent, generate_valid_actions, state_as_ordered_string
from netsecgame.game_components import AgentRole

class SARSAAgent(BaseAgent):

    def __init__(self, host, port, role=AgentRole.Attacker, alpha=0.1, gamma=0.6, epsilon=0.1) -> None:
        super().__init__(host, port, role)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {}
        self._str_to_id = {}

    def store_q_table(self,filename):
        with open(filename, "wb") as f:
            data = {"q_table":self.q_values, "state_mapping": self._str_to_id}
            pickle.dump(data, f)

    def load_q_table(self,filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.q_values = data["q_table"]
            self._str_to_id = data["state_mapping"]

    def get_state_id(self, state:GameState) -> int:
        state_str = state_as_ordered_string(state)
        if state_str not in self._str_to_id:
            self._str_to_id[state_str] = len(self._str_to_id)
        return self._str_to_id[state_str]
      
    def select_action(self, state:GameState, testing=False) -> Action:
        actions = generate_valid_actions(state)
        state_id = self.get_state_id(state)
        
        #logger.info(f'The valid actions in this state are: {[str(action) for action in actions]}')
        if random.uniform(0, 1) <= self.epsilon and not testing:
            action = random.choice(list(actions))
            if (state_id, action) not in self.q_values:
                self.q_values[state_id, action] = 0
            return action, state_id
        else: #greedy play
            #select the acion with highest q_value
            tmp = dict(((state_id,action), self.q_values.get((state_id,action), 0)) for action in actions)
            state_id, action = max(tmp, key=tmp.get)
            #if max_q_key not in self.q_values:
            if (state_id, action) not in self.q_values:
                self.q_values[state_id, action] = 0
            return action, state_id
        
    def play_episode(self, testing=False)->list:
        observation = self.request_game_reset()
        episodic_returns = []
        action1 ,state_id1 = self.select_action(observation.state, testing)
        done = observation.end
        while not done:
            # get next state
            observation2 = self.make_step(action1)
            episodic_returns.append(observation2.reward)
            # get action in the next state
            action2, state_id2 = self.select_action(observation2.state, testing)
            
            # use it to update the Q table
            if not testing:
                self.q_values[state_id1, action1] += self.alpha*(observation2.reward+ self.gamma*self.q_values[state_id2, action2]-self.q_values[state_id1, action1])

            # move1 step
            action1 = action2
            state_id1= state_id2
            done = observation2.end
        return episodic_returns
    
    def play_game(self, num_episodes=1, testing=False):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        
        _ = self.register()
        if not args.test_only:
            for episode in range(num_episodes):
                self.play_episode(testing=False)
                self._logger.debug(f"Episode {episode} finished. |Q_table| = {len(self.q_values)}")
                if episode and episode % args.eval_each == 0:
                    testing_returns = []
                    for _ in range(args.eval_for):
                        testing_returns.append(np.sum(self.play_episode(testing=True)))
                    self._logger.info(f"Eval after {episode} episodes: ={np.mean(testing_returns)}±{np.std(testing_returns)}")
                if episode % args.store_models_every == 0 and episode != 0:
                    self.store_q_table(f'sarsa_agent_marl.experiment{args.experiment_id}-episodes-{episode}.pickle')           
        returns = []
        for _ in range(args.eval_for):
            returns.append(np.sum(self.play_episode(testing=True)))
        self._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        self._logger.info("Terminating interaction")
        self.terminate_connection()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of testing episodes", default=25000, type=int)
    parser.add_argument("--eval_each", help="Sets periodic evaluation during training", default=5000, type=int)
    parser.add_argument("--eval_for", help="Sets length of periodic evaluation", default=5000, type=int)
    parser.add_argument("--epsilon", help="Sets epsilon for exploration", default=0.2, type=float)
    parser.add_argument("--gamma", help="Sets gamma for Q learing", default=0.9, type=float)
    parser.add_argument("--alpha", help="Sets alpha for learning rate", default=0.1, type=float)
    parser.add_argument("--logdir", help="Folder to store logs", default=path.join(path.dirname(path.abspath(__file__)), "logs"))
    parser.add_argument("--test_only", help="Only run testing", default=False, action='store_true')
    parser.add_argument("--experiment_id", help="Id of the experiment to record into Mlflow.", default='sarsa_006_coordinatorV3', type=str)
    parser.add_argument("--store_models_every", help="Store a model to disk every these number of episodes.", default=2000, type=int)
    parser.add_argument("--previous_model", help="Store a model to disk every these number of episodes.", type=str)
    args = parser.parse_args()

    if not path.exists(args.logdir):
        makedirs(args.logdir)
    logging.basicConfig(filename=path.join(args.logdir, "sarsa_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Create agent
    agent = SARSAAgent(args.host, args.port, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)

    if args.test_only:
        agent.load_q_table(args.previous_model)
        agent.play_game(args.episodes, testing=True)       
    else:
        agent.play_game(args.episodes, testing=False)
        agent.store_q_table("./sarsa_agent_marl.pickle")

