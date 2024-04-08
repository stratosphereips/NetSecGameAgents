from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb
import argparse


def generate_states_actions(df: pd.DataFrame) -> tuple:
    df["action"] = df["response"].map(lambda x: str(eval(x)["action"]))

    grouped = df.groupby("state").agg(list).reset_index()

    states = grouped["state"].to_list()
    actions = grouped["action"].to_list()

    metadata = [{"action": "|".join(action)} for action in actions]

    return states, metadata


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

    model = SentenceTransformer(args.embeddings_model)

    df = pd.read_csv(args.replay_buffer)
    df2 = df.query("evaluation > 5")[["state", "response"]]
    df2 = df2.sample(frac=1, random_state=42)

    # TODO: Handle states that are duplicates. Right now only the first one will be inserted I think (?)
    states, action_metadata = generate_states_actions(df2)
    print(action_metadata[:10])
    assert len(states) == len(action_metadata)

    # Generate embeddings using sentence transformers
    embeddings = model.encode(states)

    client = chromadb.PersistentClient(path="embeddings")
    # get a collection or create if it doesn't exist already
    collection = client.get_or_create_collection(
        "states", metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        embeddings=embeddings,
        metadatas=action_metadata,
        documents=["doc" + str(i) for i in range(len(embeddings))],
        ids=["state" + str(i) for i in range(len(embeddings))],
    )
