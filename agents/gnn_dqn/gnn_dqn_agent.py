# Authors:  Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz    
import numpy as np
import argparse
import os
import sys
import tensorflow_gnn as tfgnn
import tensorflow as tf
from tensorflow_gnn.models.gcn import gcn_conv
import random

# This is used so the agent can see the environment and game components
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) )))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))

#with the path fixed, we can import now
from env.game_components import GameState, Action
from base_agent import BaseAgent
from graph_agent_utils import state_as_graph
from agent_utils import generate_valid_actions

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class GnnDQNAgent(BaseAgent):
    """
    Class implementing the DQN algorithm with GNN as input layer
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args.host, args.port, args.role)
        self.args = args
        graph_schema = tfgnn.read_schema(os.path.join(path.dirname(path.abspath(__file__)),"./schema.pbtxt"))
        self._example_input_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
        run_name = "netsecgame__GNN_DQN"
        self._tf_writer = tf.summary.create_file_writer("./logs/"+ run_name)
        self.replay_memory = []
        self._MAE_metric = tf.keras.metrics.MeanAbsoluteError()
        
        register_obs = self.register()
        num_actions = register_obs.info["num_actions"]
        self.actions_idx = {}
        self.idx_to_action = {}
        
        def set_initial_node_state(node_set, node_set_name):
            d1 = tf.keras.layers.Dense(128,activation="relu")(node_set["node_type"])
            return tf.keras.layers.Dense(64,activation="relu")(d1)

        #input
        input_graph = tf.keras.layers.Input(type_spec=self._example_input_spec, name="input")
        #process node features with FC layer
        graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state, name="node_preprocessing")(input_graph)
        # Graph convolution
        for i in range(args.graph_updates):
            graph = gcn_conv.GCNHomGraphUpdate(units=128, add_self_loops=True, name=f"GCN_{i+1}")(graph)
        #node_emb = tfgnn.keras.layers.Readout(node_set_name="nodes")(graph)
        pooling = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "mean",node_set_name="nodes", name="pooling")(graph)
        # Two hidden layers (Following the DQN)
        hidden1 = tf.keras.layers.Dense(128, activation="relu", name="hidden1")(pooling)
        hidden2 = tf.keras.layers.Dense(64, activation="relu", name="hidden2")(hidden1)
        output = tf.keras.layers.Dense(num_actions, activation=None, name="output")(hidden2)
        self._model = tf.keras.Model(input_graph, output, name="Q1")
        self._model.compile(tf.keras.optimizers.Adam(learning_rate=args.lr))
        self._target_model = tf.keras.models.clone_model(self._model)
        self._model.summary()

    def _build_batch_graph(self, state_graphs):
        print(f"Building scalar graphs from {len(state_graphs)} states")
        def _gen_from_list():
            for g in state_graphs:
                yield g
        ds = tf.data.Dataset.from_generator(_gen_from_list, output_signature=self._example_input_spec)
        graph_tensor_batch = next(iter(ds.batch(self.args.batch_size)))
        scalar_graph_tensor = graph_tensor_batch.merge_batch_to_components()
        return scalar_graph_tensor
    
    def state_to_graph_tensor(self, state:GameState):
        X,E = state_as_graph(state)

        src,trg = [x[0] for x in E],[x[1] for x in E]
        
        graph_tensor =  tfgnn.GraphTensor.from_pieces(
                node_sets = {"nodes":tfgnn.NodeSet.from_fields(
                    sizes = [X.shape[0]],
                    features = {"node_type":X}
                )},
                edge_sets = {
                    "related_to": tfgnn.EdgeSet.from_fields(
                    sizes=[len(src)],
                    features = {},
                    adjacency=tfgnn.Adjacency.from_indices(
                    source=("nodes", np.array(src, dtype='int32')),
                    target=("nodes", np.array(trg, dtype='int32'))))
                }
            )
        return graph_tensor

    def predict_action(self, state, training=False)-> Action:
        valid_actions = self.get_valid_actions(state)
        # random action
        if training and np.random.uniform() < self.args.epsilon:
            return random.choice(valid_actions)
        else:
            valid_action_indices = tf.constant([self.actions_idx[a] for a in valid_actions], dtype=tf.int32)
            # greedy action
            state_graph = self.state_to_graph_tensor(state)
            model_output = tf.squeeze(self._model(state_graph))
            max_idx = np.argmax(tf.gather(model_output, valid_action_indices))
            return self.idx_to_action[max_idx]

    def _make_training_step(self, inputs, actions, y_true)->None:
        #perform training step
        with tf.GradientTape() as tape:
            y_hat = tf.gather_nd(self._model(inputs, training=True), actions)
            mse = tf.keras.losses.MeanSquaredError()
            loss = mse(y_true, y_hat)
        grads = tape.gradient(loss, self._model.trainable_weights)
        #grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self._model.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
        self._MAE_metric.update_state(y_true, y_hat)
        with self._tf_writer.as_default():
            tf.summary.scalar('train/MSE_Q', loss, step=self._model.optimizer.iterations)

    def process_rewards(self, transition:tuple)->float:
        _,_,r,s_next,end = transition
        if end:
            return r #terminal state, return just the reward
        else:
            valid_actions = self.get_valid_actions(s_next)
            valid_action_indices = np.array([self.actions_idx[a] for a in valid_actions])
            state_graph = self.state_to_graph_tensor(s_next)
            model_output = tf.squeeze(self._target_model(state_graph))
            return r + args.gamma*np.max(tf.gather(model_output, valid_action_indices))
    
    def get_valid_actions(self, state:GameState):
        valid_actions = generate_valid_actions(state)
        for a in valid_actions:
            if a not in self.actions_idx:
                idx = len(self.actions_idx)
                self.actions_idx[a] = idx
                self.idx_to_action[idx] = a
        return valid_actions
    
    def train(self):
        self._MAE_metric.reset_state()
        for episode in range(self.args.episodes):
            observation = self.request_game_reset()
            while not observation.end:
                current_state = observation.state
                action = self.predict_action(current_state, training=True)
                next_observation = self.make_step(action)
                reward = next_observation.reward
                self.replay_memory.append((current_state, action, reward, next_observation.state, next_observation.end))
                observation = next_observation
            # Enough data collected, lets train
            if len(self.replay_memory) >= self.args.batch_size:
                batch_transitions = random.sample(self.replay_memory, self.args.batch_size)
                batch_inputs = self._build_batch_graph([self.state_to_graph_tensor(t[0]) for t in batch_transitions])
                batch_actions = np.array([[i, self.actions_idx[t[1]]] for i,t in enumerate(batch_transitions)])
                batch_targets = np.array([self.process_rewards(t) for t in batch_transitions])
                self._make_training_step(batch_inputs, batch_actions, batch_targets)
            # update target model
            if episode > 0 and episode % self.args.update_target_each == 0:
                self._target_model.set_weights(self._model.get_weights())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--role", help="Role of the agent", default="Attacker", type=str, action='store', required=False)

    #model arguments
    parser.add_argument("--gamma", help="Sets gamma for discounting", default=0.9, type=float)
    parser.add_argument("--batch_size", help="Batch size for NN training", type=int, default=32)
    parser.add_argument("--epsilon", help="Batch size for NN training", type=int, default=0.2)
    parser.add_argument("--graph_updates", help="Number of GCN passes", type=float, default=3)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    
    #training arguments
    parser.add_argument("--episodes", help="Sets number of training episodes", default=100, type=int)
    parser.add_argument("--eval_each", help="During training, evaluate every this amount of episodes.", default=100, type=int)
    parser.add_argument("--eval_for", help="Sets evaluation length", default=250, type=int)
    parser.add_argument("--final_eval_for", help="Sets evaluation length", default=1000, type=int)
    parser.add_argument("--update_target_each", help="Set how often should be target model updated", type=int, default=10)


    args = parser.parse_args()
    args.filename = "GNN_DQN_Agent_" + ",".join(("{}={}".format(key, value) for key, value in sorted(vars(args).items()) if key not in ["evaluate", "eval_each", "eval_for"])) + ".pickle"
    agent = GnnDQNAgent(args)
    agent.train()
   