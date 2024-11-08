#Set Up Vector DB Based On Increamental Data.

import psycopg2
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone


class DatabaseManager:
    def __init__(self, host, username, password, dbname, port):
        self.host = host
        self.username = username
        self.password = password
        self.dbname = dbname
        self.port = port
        self.connection = self.connect_to_db()

    def connect_to_db(self):
        try:
            conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.username,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Successfully connected to the database.")
            return conn
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")
            raise

    def fetch_schema_with_data_types(self):
        try:
            query = """
            SELECT table_name, column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND (data_type = 'character varying' OR data_type IN ('text', 'varchar'))
            ORDER BY table_name;
            """
            schema_df = pd.read_sql(query, self.connection)
            print("Fetched schema with data types successfully.")
            return schema_df
        except Exception as e:
            print(f"Error fetching schema: {e}")
            raise

    def fetch_unique_values_by_modified_date(self, table_name, column_name, modified_columns):
        # Get the current date and time in MM-DD-YYYY HH:MM format
        current_datetime = datetime.now().strftime('%m-%d-%Y %H:%M')
        print(f"Fetching unique values for {column_name} in {table_name} for date: {current_datetime}")
        
        # Check which modified columns exist in the current table schema
        existing_modified_columns = [
            col for col in modified_columns if col in self.fetch_schema_with_data_types().query(f'table_name == "{table_name}"')['column_name'].values
        ]
        
        if not existing_modified_columns:
            print(f"No valid modified columns found for table: {table_name}")
            return []  # Return empty if no valid column is found
    
        for modified_column in existing_modified_columns:
            try:
                query = f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                WHERE TO_CHAR("{modified_column}", 'MM-DD-YYYY HH24:MI') = %s;
                """
                df = pd.read_sql(query, self.connection, params=(current_datetime,))
                print(f"Fetched unique values for {column_name}: {df[column_name].tolist()}")
                return df[column_name].dropna().astype(str).tolist()
            except Exception as e:
                print(f"Error fetching unique values for {column_name} in {table_name} using {modified_column}: {e}")
                continue
        return []  # Return empty if no valid column is found
    

class EmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.model = self.load_huggingface_model(model_name)

    def load_huggingface_model(self, model_name):
        print(f"Loading model: {model_name}")
        return SentenceTransformer(model_name)

    def generate_embeddings(self, unique_values):
        try:
            embeddings = self.model.encode(unique_values)
            print(f"Generated embeddings for {len(unique_values)} unique values.")
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []


class PineconeManager:
    def __init__(self, api_key, index_name, batch_size=200):
        self.api_key = api_key
        self.index_name = index_name
        self.batch_size = batch_size
        self.index = self.initialize_pinecone()

    def initialize_pinecone(self):
        pc = Pinecone(api_key=self.api_key)
        index = pc.Index(self.index_name)
        print(f"Pinecone index '{self.index_name}' initialized.")
        return index

    def check_if_exists(self, table_name, column_name, unique_value):
        vector_id = f"{table_name}_{column_name}_{unique_value}"
        try:
            result = self.index.fetch([vector_id], namespace=table_name)
            exists = bool(result["vectors"])
            print(f"Vector existence check for {vector_id}: {exists}")
            return exists
        except Exception as e:
            print(f"Error checking existence in Pinecone: {e}")
            return False

    def batch_embeddings(self, upsert_data):
        for i in range(0, len(upsert_data), self.batch_size):
            yield upsert_data[i:i + self.batch_size]

    def upsert_embeddings(self, embeddings_dict):
        for table_name, columns in embeddings_dict.items():
            for column_name, data in columns.items():
                upsert_data = []
                for i, embedding in enumerate(data['embeddings']):
                    unique_value = data['unique_values'][i]

                    # Check if the vector already exists
                    if self.check_if_exists(table_name, column_name, unique_value):
                        print(f"Vector for {unique_value} already exists. Skipping.")
                        continue

                    vector_id = f"{table_name}_{column_name}_{i}"
                    metadata = {"column_name": column_name, "unique_value": unique_value}

                    upsert_data.append({
                        "id": vector_id,
                        "values": embedding.tolist(),
                        "metadata": metadata
                    })

                # Batch the upsert to avoid exceeding size limits
                for batch in self.batch_embeddings(upsert_data):
                    self.index.upsert(vectors=batch, namespace=table_name)
                    print(f"Upserted batch for {column_name} in {table_name}")


class DataProcessor:
    def __init__(self, db_manager, embedding_generator, pinecone_manager):
        self.db_manager = db_manager
        self.embedding_generator = embedding_generator
        self.pinecone_manager = pinecone_manager

        # Possible last modified column names
        self.modified_columns = ['last_modified_time', 'last_updated_time', 'last_modified_date']

    def process_data(self):
        schema_df = self.db_manager.fetch_schema_with_data_types()

        # Verify which modified columns exist in the schema
        existing_modified_columns = []
        for modified_column in self.modified_columns:
            if modified_column in schema_df['column_name'].values:
                existing_modified_columns.append(modified_column)

        print(f"Using modified columns: {existing_modified_columns}")

        embeddings_dict = {}
        for table_name in schema_df['table_name'].unique():
            embeddings_dict[table_name] = {}
            table_columns = schema_df[schema_df['table_name'] == table_name]

            for column_name in table_columns['column_name']:
                unique_values = self.db_manager.fetch_unique_values_by_modified_date(
                    table_name, column_name, existing_modified_columns)

                if unique_values:
                    embeddings = self.embedding_generator.generate_embeddings(unique_values)
                    embeddings_dict[table_name][column_name] = {
                        "unique_values": unique_values,
                        "embeddings": embeddings
                    }

        self.pinecone_manager.upsert_embeddings(embeddings_dict)


if __name__ == "__main__":
    # Initialize Database, Embedding Generator, and Pinecone Manager
    db_manager = DatabaseManager(
        host="database-test-postgress-instance.cpk2uyae6iza.ap-south-1.rds.amazonaws.com",
        username="python_aiml",
        password="python",
        dbname="python_test_poc",
        port=5432
    )
    
    embedding_generator = EmbeddingGenerator()
    
    pinecone_manager = PineconeManager(
        api_key="7844b232-6ba4-4aef-884b-7f826ec88d74",
        index_name="jagoai"
    )

    # Create DataProcessor to orchestrate the process
    data_processor = DataProcessor(db_manager, embedding_generator, pinecone_manager)
    data_processor.process_data()
