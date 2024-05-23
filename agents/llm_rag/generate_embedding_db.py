# from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

import pandas as pd
import chromadb
import argparse
import json

# This is used so the agent can see the BaseAgent
import sys
from os import path

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from env.game_components import GameState

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from llm_utils import create_status_from_state


def generate_states_actions(df: pd.DataFrame) -> tuple:
    df["action"] = df["response"].map(lambda x: str(eval(x)["action"]))

    grouped = df.groupby("state").agg(list).reset_index()

    states = grouped["state"].to_list()
    actions = grouped["action"].to_list()

    metadata = [{"action": "|".join(action)} for action in actions]

    states_str = [create_status_from_state(GameState.from_json(st)) for st in states]

    return states_str, metadata


def generate_states_actions2(df: pd.DataFrame) -> tuple:
    # df["action"] = df["response"].map(lambda x: str(eval(x)["action"]))

    # grouped = df.groupby("state").agg(list).reset_index()

    # states = grouped["state"].to_list()
    states = df["state"].to_list()
    # actions = grouped["action"].to_list()
    actions = df["response"].to_list()

    # metadata = [{"action": "|".join(action)} for action in actions]
    metadata = [{"action": action} for action in actions]

    states_str = []
    for i, st in enumerate(states):
        try:
            # print(i + 2, json.loads(st))
            status_str = create_status_from_state(GameState.from_json(st))
            states_str.append(status_str)
        except:
            print(f"problematic row {i}, {st}")
    # states_str = [create_status_from_state(GameState.from_json(st)) for st in states]

    return states_str, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_model",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="LLM used to create embeddings",
    )
    parser.add_argument("--database_folder", type=str, default="embeddings_db")
    parser.add_argument("--replay_buffer", type=str, required=True)

    args = parser.parse_args()

    # model = SentenceTransformer(args.embeddings_model)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embeddings_model
    )

    df = pd.read_csv(args.replay_buffer)
    df2 = df.query("evaluation > 5")[["state", "response"]]
    df2 = df2.sample(frac=1, random_state=42)

    # TODO: Handle states that are duplicates. Right now only the first one will be inserted I think (?)
    states, action_metadata = generate_states_actions2(df2)
    print(action_metadata[:10])
    assert len(states) == len(action_metadata)

    # Generate embeddings using sentence transformers
    # embeddings = model.encode(states)
    embeddings = sentence_transformer_ef(states)

    client = chromadb.PersistentClient(path=args.database_folder)
    # get a collection or create if it doesn't exist already
    collection = client.get_or_create_collection(
        "states", metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        embeddings=embeddings,
        metadatas=action_metadata,
        # documents=["doc" + str(i) for i in range(len(embeddings))],
        documents=states,
        ids=["state" + str(i) for i in range(len(embeddings))],
    )
