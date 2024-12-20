import openai
import psycopg2
import pandas as pd
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pinecone
import os
from fuzzywuzzy import fuzz
from datetime import datetime
import spacy
from pinecone import Pinecone, ServerlessSpec
import spacy
nlp = spacy.load("en_core_web_sm")
import warnings
warnings.filterwarnings("ignore")
import configparser
class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    # OpenAI settings
    OPENAI_API_KEY = config['openai']['api_key']
    openAI_model = config['openai']['model']

    # Database settings
    DATABASE_HOST = config['database']['host']
    DATABASE_USERNAME = config['database']['username']
    DATABASE_PASSWORD = config['database']['password']
    PORT = config.getint('database', 'port')

    # Pinecone settings
    PINECONE_API_KEY = config['pinecone']['api_key']
    INDEX_NAME = config['pinecone']['index_name']

    # Model settings
    MODEL_NAME = config['model']['name']

class Initialize_config:
    def __init__(self):
        self.embedding_model=SentenceTransformer(Config.MODEL_NAME)
        self.pinecone_index=None
        self.openai_model=None
        self.prompt_template = None
    def __del__(self):
        pass
    def assign_pinecone_index(self):
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        if Config.INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=Config.INDEX_NAME,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )
        self.pinecone_index=pc.Index(Config.INDEX_NAME)
    def process_openAI_model(self):
        openai.api_key=Config.OPENAI_API_KEY
        self.openai_model = ChatOpenAI(
        openai_api_key=openai.api_key,
        model_name=Config.openAI_model,
        temperature=0.7,
        max_tokens=150)
    def set_prompt_template(self):
        template = """## Knowledge Base:
            {knowledge_base}

            ## Database Schema:
            {database_schema}

            ## Question:
            {question}

            ## Answer:
            """
        self.prompt_template=ChatPromptTemplate.from_template(template)
        
class DB_Manager:
    def __init__(self):
        self.connection = None
        self.results=''
    def __del__(self):
        pass

    def connect(self,DATABASE_DB):
        try:
            self.connection = psycopg2.connect(
                dbname=DATABASE_DB,
                user=Config.DATABASE_USERNAME,
                password=Config.DATABASE_PASSWORD,
                host=Config.DATABASE_HOST,
                port=Config.PORT
            )
            return self.connection
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")
            raise
    def close_connection(self):
        self.connection.close()
    def execute_sql_query(self,sql_query):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql_query)
                results = cursor.fetchall()
                self.results=results
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            self.results=[]
        
class Schema_manager:
    def __init__(self,conn,query,schema):
        self.conn=conn
        self.query=query
        self.schema_df=schema
        self.schema_str = ""
    def __del__(self):
        pass
    def fetch_schema_with_data_types(self):
        try:
            self.schema_df=pd.read_sql(self.query, self.conn)
        except Exception as e:
            print(f"Error fetching schema with data types: {e}")
            raise
    def format_schema(self):
        self.schema_str = ""
        grouped = self.schema_df.groupby('table_name')
        for table_name, group in grouped:
            columns = ', '.join([f"{row['column_name']} ({row['data_type']})" for _, row in group.iterrows()])
            self.schema_str += f"{table_name}: {columns}\n"
class Determine_querry_type:
    def __init__(self,user_query,schema_df, threshold=75):
        self.query_type="knowledge"
        self.user_query=user_query
        self.schema_df=schema_df
        self.threshold=threshold
    def __del__(self):
        pass
    def determine_query_type(self):
        user_query_lower = self.user_query.lower()
        table_names = self.schema_df['table_name'].str.lower().unique()
        column_names = self.schema_df['column_name'].str.lower().unique()
        
        # Function to check fuzzy match
        def is_fuzzy_match(query, options):
            for option in options:
                if fuzz.partial_ratio(query, option) >= self.threshold:
                    return True
            return False
        
        # Check if user query matches any table or column name
        if is_fuzzy_match(user_query_lower, table_names) or \
           is_fuzzy_match(user_query_lower, column_names):
            self.query_type="database"
        else:
            self.query_type="knowledge"
    def is_entity_present(self,schema_entities,user_text):
        user_text_lower = user_text.lower()
        for entity in schema_entities:
            if entity.lower() in user_text_lower:
                return True  
        return False
    def contains_date_related_text(self,user_input):
        # Current year and month for comparison
        current_year = datetime.now().year
        current_month = datetime.now().strftime("%B")
        
        # Patterns to match date, month, year, and relative terms
        date_pattern = r'\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})\b'
        month_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|' \
                        r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
        year_pattern = r'\b(19|20)\d{2}\b'
        time_pattern = r'\b\d+\s*(or\s*(more|fewer|less))?\s*(days?|weeks?|months?|years?)\b'
        relative_terms_pattern = r'\b(this month|this year|last month|last year|next month|next year)\b'
        # Find matches
        if (re.search(date_pattern, user_input) or 
            re.search(month_pattern, user_input, re.IGNORECASE) or 
            re.search(year_pattern, user_input) or 
            re.search(time_pattern, user_input) or
            re.search(relative_terms_pattern, user_input, re.IGNORECASE)):
            return True
        return False
    def named_entity_recognition(self,text):
        doc = nlp(text)
        # Check for entities that are either PERSON, ORG, or GPE
        for entity in doc.ents:
            if entity.label_ in ("PERSON", "ORG", "GPE"):
                return True
        return False
class Pinecone_manager:
    def __init__(self, schema_df):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames = {}
        self.searched_cols = []
        self.augmented_input = ''
        self.schema_df = schema_df
        self.extracted_Features = None
        self.cleaned_feature_dict = None
        self.embedding_model = None
        self.pinecone_index = None

    def clear_all(self):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames = {}
        self.searched_cols = []
        self.augmented_input = ''

    def process_user_input(self, user_input):
        self.extracted_Features = OpenAI_manager.extract_features_with_openai(OpenAI_manager, user_input, self.schema_df)

    def process_extracted_features(self):
        def clean_extracted_features(feature_dict):
            # Remove any keys with None or empty values
            cleaned_feature_dict = {k: v for k, v in feature_dict.items() if v}
            # Extract the non-null values into a list
            feature_list = list(cleaned_feature_dict.values())
            return cleaned_feature_dict, feature_list

        try:
            # Remove the "## Solution:" part and any other non-JSON text
            json_match = re.search(r'\{.*\}', self.extracted_Features, re.DOTALL)
            
            if json_match:
                # Extract the JSON part from the matched result
                cleaned_features = json_match.group(0)

                # Convert JSON string to a Python dictionary
                feature_dict = json.loads(cleaned_features)

                # Clean feature dictionary and feature list to remove nulls and empty values
                self.cleaned_feature_dict, feature_list = clean_extracted_features(feature_dict)

                # Return cleaned JSON and feature list
                return json.dumps(self.cleaned_feature_dict, indent=4), feature_list
            else:
                return None, []
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing features: {e}")
            return None, []

    def extract_namespace(self):
        for key in self.extracted_dict.keys():
            self.NAMESPACE.append(key)
            self.columnnames[key] = extracted_dict[key]

    def call_query_pinecone(self, user_input, p_i, e_model):
        self.pinecone_index = p_i
        self.embedding_model = e_model
        for key, val in self.cleaned_feature_dict.items():
            columns = list(val.keys())
            if self.augmented_input == '':
                self.augmented_input = self.query_pinecone_and_augment_input(user_input, key, columns)
            else:
                self.augmented_input = self.query_pinecone_and_augment_input(self.augmented_input, key, columns)
        return self.augmented_input

    def query_pinecone_and_augment_input(self, user_input, namespace, columns):
        self.augmented_input = user_input
        
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_entities = flatten_dict(self.cleaned_feature_dict)
        for column_name in columns:
            if column_name not in self.searched_cols:
                self.searched_cols.append(column_name)

                # Obtain the entity value corresponding to the current column
                entity_value = self.cleaned_feature_dict[namespace].get(column_name, None)
                if not entity_value:
                    continue  # Skip to the next column if no value is found

                # Generate the query embedding for the entity value
                query_embedding = self.embedding_model.encode([entity_value])[0]
                query_embedding = np.array(query_embedding, dtype=np.float32)

                try:
                    result = self.pinecone_index.query(
                        namespace=namespace,
                        vector=query_embedding.tolist(),
                        filter={"column_name": {"$eq": column_name}},
                        top_k=3,
                        include_values=True,
                        include_metadata=True
                    )

                    matches = result.get('matches', [])
                    if matches:
                        # Automatically select the best match (first match)
                        best_match = matches[0]['metadata'].get('unique_value', entity_value)
                        self.augmented_input = self.augmented_input.replace(entity_value, best_match)
                    else:
                        print(f"No matches found for {entity_value} in Pinecone.")
                except Exception as e:
                    print(f"Error querying Pinecone: {str(e)}")

        return self.augmented_input

                              
class OpenAI_manager:
    def __init__(self):
        self.sql_query=''
        self.sql_response=''
        self.response=''
        self.prompt_template = ChatPromptTemplate.from_template(template = """
        ## Knowledge Base:
        {knowledge_base}

        ## Database Schema:
        {database_schema}

        ## Question:
        {question}

        ## Answer:
        """)
    def __del__(self):
        pass
    def get_answer_from_chatbot(self,question, database_schema,openai_model):
        try:
            prompt = self.prompt_template.format(
                knowledge_base="",
                database_schema=database_schema,
                question=question
            )
            response = openai_model.invoke(input=prompt)
            parsed_response = response.content.strip() if hasattr(response, 'content') else "No response content found."
            print("Resp:",parsed_response)
            return parsed_response
        except Exception as e:
            return f"Error generating response from OpenAI: {str(e)}"  
        
    def extract_features_with_openai(self,user_input, processed_schema_df):
        schema_json = processed_schema_df.to_json(orient='records')

        # Define aggregation keywords
        aggregation_keywords = ["count", "sum", "average", "min", "max"]

        # Check if the user input contains aggregation keywords
        contains_aggregation = any(keyword in user_input.lower() for keyword in aggregation_keywords)

        # Determine if the input is a general list query
        general_query_keywords = ["list all", "show", "retrieve", "get all", "all projects"]
        is_general_query = any(keyword in user_input.lower() for keyword in general_query_keywords)

        # If it's a general query, return an empty response or a structured response indicating it's a general query
        if is_general_query:
            return '{"projects": {}}'  # or return some structured message indicating this is a general list request

        # Enhanced prompt to better understand user intent and accurately extract relevant entities
        prompt = f"""
        ## Database Schema Context:
        The database contains the following schema (tables and columns with their data types):
        {schema_json}

        ## User Input:
        The user provided the following input: "{user_input}"

        ## Task:
        Your task is to deeply understand the user's intent and extract relevant features, values, and table names from the user input based on the schema. Specifically:
        
        1. **User Intent**:
        - First, analyze the input to infer whether the user is asking for **specific details**, performing **search queries** (e.g., filtering on values), or seeking **aggregated results** (e.g., counts, sums).
        - Accurately distinguish between **list or overview queries** and **specific queries**.

        2. **Entity and Feature Extraction**:
        - Extract features, values, and entities based on the schema, focusing on:
            - Column values from varchar, char, or text fields.
            - If aggregation terms like "{', '.join(aggregation_keywords)}" are detected, ensure the correct numeric columns are considered alongside the varchar columns.
            - If specific project names or user names are detected, extract those entities precisely.

        3. **Handling Partial and Flexible Matches**:
        - Handle partial matches, spelling variations, and incomplete inputs intelligently.

        4. **Schema-based Filtering**:
        - For varchar, text, and char columns, extract relevant string values.

        5. **Output**:
        - Return a structured JSON dictionary, where:
            - Each table name corresponds to the fields and their extracted values.
            - Omit any fields or tables where no relevant values were found.
        
        Example JSON output format:
        {{
            "projects": {{
            "project_name": "IIFL Tower"
            }},
            "users": {{
            "owner": "John Doe"
            }}
        }}
        """

        try:
            # Call OpenAI to extract features/entities using the refined prompt
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are an expert in extracting entities from user input by mapping them to relevant database schema."}, 
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.5
            )
            
            # Extract the response text
            extracted_features = response.choices[0].message.content.strip()
            return extracted_features
        except openai.OpenAIError as e:
            print(f"Error with OpenAI: {e}")
            raise
    def generate_sql_query(self, processed_schema_str, aug_input):
        prompt = f"""
        You are an expert in SQL query generation, specialized in mapping user input to relevant database tables and columns.

        The database has the following schema structure:
        {processed_schema_str}

        The user provided this input:
        "{aug_input}"

        Your task is to generate an **optimized and accurate SQL query** that adheres to the following principles:
        1. **User Intent**:
        - Understand the user's intent thoroughly. For example, identify whether they are asking for a **list** or **specific details**.
        - Handle both explicit and implicit queries (e.g., incomplete phrases, partial project names, or ambiguous terms).
        
        2. **Schema and Data Relationships**:
        - Map the user’s input to the correct **tables** and **columns** based on the schema.
        - Ensure that foreign key relationships are correctly used if multiple tables are involved (e.g., joins).

        3. **SQL Operators**:
        - For partial matches, use `LIKE` with wildcards (`%`) appropriately, ensuring flexible handling of partial inputs or spelling variations.
        - Use **appropriate operators** based on the input (e.g., `=`, `>`, `IN`, etc.).

        4. **Performance Optimization**:
        - If the user does not specify the number of results, apply a default `LIMIT 10` for performance.
        - Use **indexed columns** where possible to speed up queries.
        - Apply `ORDER BY` if the user implies sorting (e.g., "latest projects", "top-rated projects").

        5. **Data Type Handling**:
        - Handle **data type mismatches** by casting where necessary (e.g., comparing strings to dates or numbers).
        - Be cautious of null values or missing data that could impact query results.

        6. **Case Sensitivity**:
        - Maintain **case sensitivity** in string comparisons but handle user input variations without converting to lowercase.
        - Ensure that exact matches honor the original casing in the database, but flexible enough for input variations.

        7. **Handling Errors and Ambiguity**:
        - If the user input is ambiguous or contains partial project names, ensure the query handles these cases and selects the most relevant match from the data (e.g., fuzzy matching).

        Output the generated SQL query in a **well-formatted** and **executable** way.

        """
        # Call GPT-4o-mini-2024-07-18 model using chat completion API
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in generating optimized SQL queries based on user input and database schema."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Token limit for generated completion
            temperature=0.7  # Slight temperature for creative output
        )

        # Extract SQL query from the response
        sql_response = response.choices[0].message.content
        self.sql_response = sql_response

    # Find and clean the SQL query part
    start = sql_response.find("```sql") + 6
    end = sql_response.find("```", start)
    self.sql_query = sql_response[start:end].strip()
    print(self.sql_query)

    def generate_response(self,aug_input,results):
        # Prepare the prompt for GPT-4 to generate the natural language response
        prompt = f"User query: \"{aug_input}\"\nSQL result: {results}\nGenerate a natural language response from the result:"
        
        # Call the OpenAI Chat API
        response = openai.chat.completions.create(
          model="gpt-4o-mini-2024-07-18",
          messages=[
              {"role": "user", "content": prompt}
          ],
          max_tokens=500,
          temperature=0.7
        )
        
        self.response=response.choices[0].message.content     
user_input=''

def main(db_name='',schema='',key='',data=''):
    global user_input,DB,p,conn,openai_manager
    # Initialize handlers
    selection=None
    query = f"""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{schema}'
            """
    if schema:
        schema_manager=Schema_manager(conn,query,schema)
        schema_manager.fetch_schema_with_data_types()
        schema_manager.format_schema()
        determine_querry=Determine_querry_type(user_input,schema_manager.schema_df)
        determine_querry.determine_query_type()
        print(determine_querry.query_type)
        if determine_querry.query_type=="database":
            is_entity=determine_querry.is_entity_present(schema_manager.schema_df['column_name'].tolist(),
                                        user_input)
            if determine_querry.contains_date_related_text(user_input):
                if is_entity:
                    pine_cone=PineCone_Manager(schema_manager.schema_df)
                    pine_cone.process_user_input(user_input)
                    _, feature_list=pine_cone.process_extracted_features()
                    pine_cone.call_query_pinecone(user_input,p.pinecone_index,p.embedding_model)
                    openai_manager.generate_sql_query(schema_manager.schema_str,pine_cone.augmented_input)
                    DB.execute_sql_query(openai_manager.sql_query)
                    print(DB.results)
                    if len(DB.results)!=0:
                        openai_manager.generate_response(pine_cone.augmented_input,DB.results)
                        pine_cone.clear_all()
                        return (openai_manager.response)
                    else:
                        pine_cone.clear_all()
                        return ("I'm sorry, but I'm unable to provide results. Could you please clarify your query so I can assist you better?")
                    
                    
                else:
                    openai_manager.generate_sql_query(schema_manager.schema_str,user_input)
                    DB.execute_sql_query(openai_manager.sql_query)
                    openai_manager.generate_response(user_input,DB.results)
                    return (openai_manager.response)
            else:
                pine_cone=Pinecone_manager(schema_manager.schema_df)
                pine_cone.process_user_input(user_input)
                _, feature_list=pine_cone.process_extracted_features()
                pine_cone.call_query_pinecone(user_input,p.pinecone_index,p.embedding_model)
                
                openai_manager.generate_sql_query(schema_manager.schema_str,pine_cone.augmented_input)
                DB.execute_sql_query(openai_manager.sql_query)
                if len(DB.results)!=0:
                    openai_manager.generate_response(pine_cone.augmented_input,DB.results)
                    pine_cone.clear_all()
                    return (openai_manager.response)
                else:
                    pine_cone.clear_all()
                    return ("I'm sorry, but I'm unable to provide results. Could you please clarify your query so I can assist you better?")
                
        else:
            return (openai_manager.get_answer_from_chatbot(user_input, schema_manager.schema_str,p.openai_model))
        
    else:
        return ("No schema Selected")
    DB.close_connection()
DB,p,conn,openai_manager=None,None,None,None
user_input=''
DB=DB_Manager()
openai_manager=OpenAI_manager()
p=Initialize_config()
p.assign_pinecone_index()
p.process_openAI_model()
p.set_prompt_template()
db_name="python_test_poc_two"
conn = DB.connect(DATABASE_DB = f"{db_name}")
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.route('/process', methods=['POST'])
def process_request():
    global user_input,conn
    try:
        print(conn)
        data = request.json
        print(data)
        key=next(iter(data.keys()))
        data=data[key]
        user_input=data
        print(user_input)
        result=main(db_name="python_test_poc_two",schema='public',key=key,data=data)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"result": "There is an issue with query genration, query can not be executed with selected please provide proper query"})
@app.route('/select_db', methods=['POST'])
def assign_db():
    global db_name
    data = request.json
    key=next(iter(data.keys()))
    db_name=data[key]
    return jsonify({"result":"DB selected successfully"})
    

if __name__ == "__main__":
    app.run(host= '0.0.0.0',debug=True,port=5001)

        
    
