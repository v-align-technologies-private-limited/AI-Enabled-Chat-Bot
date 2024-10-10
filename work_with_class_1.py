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
class Config:
    OPENAI_API_KEY = ""
    DATABASE_HOST = "database-test-postgress-instance.cpk2uyae6iza.ap-south-1.rds.amazonaws.com"
    DATABASE_USERNAME = "postgres"
    DATABASE_PASSWORD = "valign#123"
    PORT = 5432
    PINECONE_API_KEY = "9fbe58e4-9e72-4023-90eb-ba8d022916b5"  
    INDEX_NAME = "smart-desk"
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    openAI_model="gpt-4o-mini-2024-07-18"
class Initialize_config:
    def __init__(self):
        self.embedding_model=SentenceTransformer(Config.MODEL_NAME)
        self.pinecone_index=None
        self.openai_model=None
        self.prompt_template = None
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
class Schema_manager:
    def __init__(self,conn,query,schema):
        self.conn=conn
        self.query=query
        self.schema_df=schema
        self.schema_str = ""
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
    def __init__(self,schema_df):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames={}
        self.searched_cols=[]
        self.augmented_input=''
        self.schema_df=schema_df
        self.extracted_Features=None
        self.cleaned_feature_dict=None
        self.pinecone_data=None
        self.embedding_model = None
        self.pinecone_index = None
    def clear_all(self):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames={}
        self.searched_cols=[]
        self.augmented_input=''
        self.pinecone_data=None
    def process_user_input(self,user_input):
        self.extracted_Features=OpenAI_manager.extract_features_with_openai(OpenAI_manager,user_input,self.schema_df)
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
            #print("match:",json_match)
            
            if json_match:
                # Extract the JSON part from the matched result
                cleaned_features = json_match.group(0)

                # Convert JSON string to a Python dictionary
                feature_dict = json.loads(cleaned_features)

                # Clean feature dictionary and feature list to remove nulls and empty values
                self.cleaned_feature_dict, feature_list = clean_extracted_features(feature_dict)
                #print("Dict:",cleaned_feature_dict)

                # Return cleaned JSON and feature list
                return json.dumps(self.cleaned_feature_dict, indent=4), feature_list
            else:
                return None, []
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing features: {e}")
            return None, []
    def extract_nmaespace(self):
        for key in self.extracted_dict.keys():
            self.NAMESPACE.append(key)
            self.columnnames[key]= extracted_dict[key]
    def call_query_pinecone(self,user_input,p_i,e_model):
        self.pinecone_index=p_i
        self.embedding_model=e_model
        for key,val in self.cleaned_feature_dict.items():
            x=list(val.keys())
            if self.augmented_input=='':
                self.query_pinecone_and_augment_input(user_input,key,x)
            else:
                self.query_pinecone_and_augment_input(self.augmented_input,key,x)
    def query_pinecone_and_augment_input(self,user_input,namespace, columns):
        self.augmented_input = user_input
        self.pinecone_data = {}
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
                        filter={
                            "column_name": {"$eq": column_name}
                        },
                        top_k=3,
                        include_values=True,
                        include_metadata=True
                    )
                    
                    matches = result.get('matches', [])
                    #print("matches:",matches)
                    if matches:
                        unique_values = [match['metadata'].get('unique_value') for match in matches if 'metadata' in match]
                        if unique_values:
                            self.pinecone_data[column_name] = unique_values
                            if len(unique_values) > 1:
                                print(f"Multiple matches found for '{entity_value}':")
                                for idx, unique_value in enumerate(unique_values):
                                    print(f"{idx + 1}: {unique_value}")
                                while True:
                                    selection = input(f"Please select the most relevant option for '{entity_value}' (1-{len(unique_values)}): ")
                                    try:
                                        selected_value = unique_values[int(selection) - 1]
                                        self.augmented_input = self.augmented_input.replace(entity_value, selected_value)
                                        break
                                    except (IndexError, ValueError):
                                        print("Invalid selection. Please choose a valid option.")
                            else:
                                self.augmented_input = self.augmented_input.replace(entity_value, unique_values[0])
                    else:
                        print(f"No matches found for {entity_value} in Pinecone.")
                except Exception as e:
                    print(f"Error querying Pinecone: {str(e)}")
                              
class OpenAI_manager:
    def __init__(self):
        pass
    def extract_features_with_openai(self,user_input, processed_schema_df):
        schema_json = processed_schema_df.to_json(orient='records')
        # Refined prompt to ensure OpenAI extracts table names, column names, and their values
        prompt = f"""
        ## Database Schema Context:
        The following represents the columns and their respective tables available in the database:
        {schema_json}

        ## User Input:
        The user has provided the following input: "{user_input}"

        ## Task:
        Extract the relevant features, values, and table names from the user input based on the schema. These features might include project names, owners, dates, statuses, etc., along with their corresponding table names.

        ## Instructions:
        - Return a JSON dictionary that includes the table names as keys, and within each table, include the fields and their values extracted from the user input.
        - Omit any fields or tables where the value is empty or null.
        - Format the output as a JSON object with keys only for tables and fields that have values.
        """

        try:
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                temperature=0.5
            )
            extracted_features = response.choices[0].text.strip()
            return extracted_features
        except openai.OpenAIError as e:
            print(f"Error with OpenAI: {e}")
            raise

        
        
def main():
    # Initialize handlers
    DB=DB_Manager()
    p=Initialize_config()
    p.assign_pinecone_index()
    p.process_openAI_model()
    p.set_prompt_template()
    db_name="python_test_poc"
    conn = DB.connect(DATABASE_DB = f"{db_name}")
    schema='public'
    query = f"""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{schema}'
            """
    user_input=input("Enter question:")
    if schema:
        schema_manager=Schema_manager(conn,query,schema)
        schema_manager.fetch_schema_with_data_types()
        schema_manager.format_schema()
        determine_querry=Determine_querry_type(user_input,schema_manager.schema_df)
        determine_querry.determine_query_type()
        if determine_querry.query_type=="database":
            is_entity=determine_querry.is_entity_present(schema_manager.schema_df['column_name'].tolist(),
                                        user_input)
            if determine_querry.contains_date_related_text(user_input):
                if is_enity:
                    pine_cone=PineCone_Manager(schema_manager.schema_df)
                    pine_cone.clear_all()
                    pine_cone.process_user_input(user_input)
                    _, feature_list=pine_cone.process_extracted_features()
                    pine_cone.call_query_pinecone(user_input,p.pinecone_index,p.embedding_model)
                    print("Aug input:",pine_cone.augmented_input)
                    print("pinecone data:",pine_cone.pinecone_data)
                    
                    
                else:
                    print("Contains date related value and No entity")
            else:
                pine_cone=Pinecone_manager(schema_manager.schema_df)
                pine_cone.clear_all()
                pine_cone.process_user_input(user_input)
                _, feature_list=pine_cone.process_extracted_features()
                pine_cone.call_query_pinecone(user_input,p.pinecone_index,p.embedding_model)
                print("Aug input:",pine_cone.augmented_input)
                print("pinecone data:",pine_cone.pinecone_data)
                
        else:
            print("No database")
        
    else:
        print("No schema")
if __name__ == "__main__":
    main()
        
    
