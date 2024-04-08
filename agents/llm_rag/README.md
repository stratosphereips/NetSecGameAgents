# RAG LLM Agent

The agent is based on the concept of Retrieval-Augmented Generation (RAG). When the agent receives a state of the environment it generates an embedding for the string representation of the state. Then, it queries the vector database to find the top-n states with similar embeddings and the actions that were taken in the respective states. 
The list of actions is then sampled and along with other information it is sent to the LLM, which has to populate it with the correct parameters.

## Vector DB
The agent requires a database of state embeddings with their respective actions. The underlying vector database is Chroma DB and the model used for the embedding generation is `mixedbread-ai/mxbai-embed-large-v1`. However, other models can also be used.

The database is generated using the replay buffers of the `llm_qa` agent with GPT-4 as a model, by running:

```
python generate_embedding_db.py --replay_buffer states_prompts_responses_1.csv --embeddings_model mixedbread-ai/mxbai-embed-large-v1
```

## Required python packages
In addition to the packages required by the `llm_qa` agent the following are also required to be present: 

```
chromadb
sentence-transformers
pandas

```

## How to run the agent

```
python llm_rag.py --llm gpt-3.5-turbo --test_episodes 1 --memory_buffer 10 --database_folder embeddings --embeddings_model mixedbread-ai/mxbai-embed-large-v1
```
