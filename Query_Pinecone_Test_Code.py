import numpy as np
import openai
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone

# Constants
PINECONE_API_KEY = "pcsk_5tGLsU_Eb1zwugguxqo8Zt1LAkL8bJihav8VyPoYdqwLBfH54A6zy9Qx4LtK6A6suh9JLq"  # Replace with your Pinecone API key
OPENAI_API_KEY = ""  # Replace with your OpenAI API key
INDEX_NAME = "jagoai"  # Replace with your index name

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Function to initialize Pinecone index
def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

# Function to generate embedding using OpenAI
def generate_openai_embedding(text):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-large",  # Correct embedding model
            input=text  # Input must be a list
        )
        embedding = response.data[0].embedding
        return embedding


    except Exception as e:
      print(f"Error generating embeddings for {text}")

# Function to query Pinecone for entities
def query_pinecone_for_entities(text):
    # Initialize Pinecone index
    pinecone_index = initialize_pinecone()

    # Dictionary to store results
    results = {}

    if text:
        try:
            # Generate embedding for the text using OpenAI
            query_embedding = generate_openai_embedding(text)
            print("query_embedding", query_embedding)


            # Query Pinecone with the specific namespace
            result = pinecone_index.query(
                namespace="projects_zoho_projects_",
                vector=query_embedding,
                top_k=3,
                include_values=True,
                include_metadata=True,
                filter={"column_name": {"$eq": "status"}}
            )
            matches = result.get("matches", [])
            print('matches', matches)
            if matches:
                # Sort matches by score in descending order
                matches.sort(key=lambda x: x["score"], reverse=True)

            else:
                print(f"No matches found for '{text}' in Pinecone.")
                results[text] = {"error": "No matches found"}

        except Exception as e:
            print(f"Error querying Pinecone for '{text}': {str(e)}")
            results[text] = {"error": str(e)}

    return results

# Static input
text = "not completed"

# Query Pinecone
result = query_pinecone_for_entities(text)
print("Query Results:")
print(result)
