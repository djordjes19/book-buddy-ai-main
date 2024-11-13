"""
Module for retrieving book summaries from a vector database using semantic similarity search.

This module provides functionality to search through book summaries stored in a ChromaDB
vector database by converting search queries into embeddings and finding the most 
semantically similar documents.
"""

import chromadb
import openai

# Name of the ChromaDB collection containing the book summaries
# This should match the collection name used when populating the database
COLLECTION_NAME = "bb_summaries"
VECTOR_DB_PATH = "../vector_db"


def retrieve(query: str, openai_client: openai.OpenAI) -> tuple[list, list]:
    """
    Retrieves the most relevant book summary for a given query using vector similarity search.

    Args:
        query (str): The search query text to find matching book summaries for
        openai_client (openai.OpenAI): Initialized OpenAI client for generating embeddings

    Returns:
        tuple[list, list]: A tuple containing:
            - contexts: List of retrieved document texts (book summaries)
            - metadata: List of metadata associated with retrieved documents

    The function performs semantic search by:
    1. Converting the query to an embedding vector using OpenAI's text-embedding-ada-002 model
    2. Using ChromaDB to find the most similar document based on cosine similarity
    3. Returning both the matched document text and its metadata
    """
    # Generate embedding vector for the search query using OpenAI's API
    # The embedding is a high-dimensional vector representing the semantic meaning
    emb = openai_client.embeddings.create(input=[query], model="text-embedding-ada-002")
    emb = emb.data[0].embedding  # Extract the actual embedding vector

    # Initialize ChromaDB client with path to persistent storage
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # Get the collection containing our book summaries and their embeddings
    data_collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Query the collection to find the most similar document
    # Uses cosine similarity between the query embedding and document embeddings
    res = data_collection.query(
        query_embeddings=[emb],
        n_results=1,  # Only retrieve the single most relevant result
    )

    # Extract and return the results
    contexts = res["documents"]  # The matched book summaries
    metadata = res["metadatas"]  # Associated metadata for the matches

    return contexts, metadata
