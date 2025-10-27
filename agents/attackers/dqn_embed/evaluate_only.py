import argparse

from ddqn_agent_black_box import DDQNAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action="store", required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action="store", required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to evaluate", default=100, type=int)
    parser.add_argument("--model_path", help="Path to the trained model", default="checkpoints/ddqn_checkpoint_best.pt", type=str)
    # parser.add_argument("--logdir", help="Folder to store logs",default=path.join(path.dirname(path.abspath(__file__)), "logs"))
    # parser.add_argument("--evaluate", help="Evaluate the agent and report, instead of playing the game only once.",action="store_true")
    # parser.add_argument("--cont", help="Continue training the final model from the previous run.", action="store_true")
    # parser.add_argument("--env_conf", help="Configuration file of the env. Only for logging purposes.", required=False, default='./env/netsecenv_conf.yaml', type=str)
    # parser.add_argument("--decay", help="Epsilon decay factor", required=False, default=1e-4, type=float)
    # parser.add_argument("--lr", help="Learning rate", required=False, default=1e-3, type=float)
    args = parser.parse_args()


    agent = DDQNAgent(args.host, args.port, "Attacker")
    # Evaluate the agent performance
    print("Evaluating the agent performance")

    observation = agent.register()
    # action_list = agent._action_list
    agent.define_networks()

    agent.request_game_reset(randomize_topology=True)
    observation = agent.request_game_reset(randomize_topology=True)
    
    # Load the final checkpoint
    agent.load(args.model_path)
    agent.q_net.eval()

    agent.eval(observation, 0, args.episodes)
    agent._logger.info("Terminating interaction")
    agent.terminate_connection()