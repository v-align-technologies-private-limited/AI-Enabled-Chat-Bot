import os
import psycopg2
import pandas as pd
import openai  # Make sure this library is installed and imported
from pinecone import Pinecone, Index, ServerlessSpec  # Adjust if using a different Pinecone client
from transformers import AutoTokenizer, AutoModel

# Database connection details
DATABASE_HOST = "database-test-postgress-instance.cpk2uyae6iza.ap-south-1.rds.amazonaws.com"
DATABASE_USERNAME = "postgres"
DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', 'valign#123')  # Use environment variable for security
DATABASE_DB = "zoho_projects_data_v2_backup"
PORT = 5432

# Pinecone details
pinecone_api_key = os.getenv('PINECONE_API_KEY', 'pcsk_5tGLsU_Eb1zwugguxqo8Zt1LAkL8bJihav8VyPoYdqwLBfH54A6zy9Qx4LtK6A6suh9JLq')  # Use environment variable for security
index_name = "jagoai"
BATCH_SIZE = 512  # Adjust the batch size to avoid exceeding the size limit

# Function to connect to PostgreSQL database
def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname=DATABASE_DB,
            user=DATABASE_USERNAME,
            password=DATABASE_PASSWORD,
            host=DATABASE_HOST,
            port=PORT
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        raise

# Fetch schema with column names and data types, only including string types
def fetch_schema_with_data_types(conn):
    try:
        query = """
        SELECT table_name, column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND (data_type = 'character varying' OR data_type IN ('text', 'varchar'))
          AND table_name IN ('projects_zoho_projects_')
        ORDER BY table_name;
        """
        schema_df = pd.read_sql(query, conn)
        print(schema_df)
        return schema_df
    except Exception as e:
        print(f"Error fetching schema with data types: {e}")
        raise

# Fetch unique values from each column along with table details
def fetch_unique_values(conn, table_name, column_name):
    try:
        query = f"SELECT DISTINCT {column_name} FROM {table_name}"
        df = pd.read_sql(query, conn)
        return df[column_name].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error fetching unique values for {column_name} in {table_name}: {e}")
        return []

# Fetch all unique values for each column and map them to table details
def fetch_all_unique_values_with_table(conn, schema_df):
    unique_values_dict = {}
    for table_name in schema_df['table_name'].unique():
        unique_values_dict[table_name] = {}
        table_columns = schema_df[schema_df['table_name'] == table_name]
        for column_name in table_columns['column_name']:
            unique_values = fetch_unique_values(conn, table_name, column_name)
            unique_values_dict[table_name][column_name] = unique_values
    return unique_values_dict

# Generate embeddings for each unique value using OpenAI's API
def generate_and_store_embeddings(unique_values_dict):
    embeddings_dict = {}
    for table_name, columns in unique_values_dict.items():
        embeddings_dict[table_name] = {}
        for column_name, unique_values in columns.items():
            if unique_values:
                unique_values = set(unique_values)
                unique_values = [item for item in unique_values if not (isinstance(item, (int, float)) or item is None or item == "")]
                embeddings = []
                processed_unique_values = []

                for val in unique_values:
                    try:
                        response = openai.embeddings.create(
                            model="text-embedding-3-large",
                            input=val
                        )
                        embedding = response.data[0].embedding

                        # Check if the embedding is valid
                        if isinstance(embedding, list) and len(embedding) == 3072 and all(isinstance(x, float) for x in embedding):
                            embeddings.append(embedding)
                            processed_unique_values.append(val)
                        else:
                            print(f"Invalid embedding for {val} in {column_name}: {embedding}")
                    except Exception as e:
                        print(f"Error generating embeddings for {val} in {column_name}: {e}")
                        continue  # Continue with the next value if error occurs

                # Store only if embeddings are successfully generated
                if embeddings:
                    embeddings_dict[table_name][column_name] = {
                        "unique_values": processed_unique_values,
                        "embeddings": embeddings
                    }
            else:
                print(f"No unique values found for {column_name} in {table_name}. Skipping embeddings.")
    return embeddings_dict

# Initialize Pinecone with Dot Product metric
def initialize_pinecone():
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    return index

# Batch the embeddings for upserts
def batch_embeddings(upsert_data, batch_size):
    for i in range(0, len(upsert_data), batch_size):
        yield upsert_data[i:i + batch_size]

# Upsert embeddings into Pinecone with metadata for each table (namespace)
def upsert_embeddings_into_pinecone(index: Index, embeddings_dict):
    for table_name, columns in embeddings_dict.items():
        for column_name, data in columns.items():
            upsert_data = []
            unique_values = data["unique_values"]
            embeddings = data["embeddings"]

            for i, (unique_value, embedding) in enumerate(zip(unique_values, embeddings)):
                if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
                    print(f"Invalid embedding for {unique_value} in {column_name}. Skipping.")
                    continue

                vector_id = f"{table_name}_{column_name}_{i}"
                metadata = {
                    "column_name": column_name,
                    "unique_value": unique_value
                }

                upsert_data.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })

            # Batch the upsert to avoid exceeding size limits
            for batch in batch_embeddings(upsert_data, BATCH_SIZE):
                try:
                    index.upsert(vectors=batch, namespace=table_name)
                    print(f"Upserted batch for {column_name} in {table_name}")
                except Exception as e:
                    print(f"Error upserting batch for {column_name} in {table_name}: {e}")

