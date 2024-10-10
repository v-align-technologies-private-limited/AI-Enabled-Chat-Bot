from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

import requests
app = Flask(__name__)

import openai
import psycopg2
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import json
from fuzzywuzzy import fuzz
import re


# Database connection details
DATABASE_HOST = "database-test-postgress-instance.cpk2uyae6iza.ap-south-1.rds.amazonaws.com"
DATABASE_USERNAME = "postgres"
DATABASE_PASSWORD = "valign#123"
DATABASE_DB = "python_test_poc"
PORT = 5432

# OpenAI API key initialization
openai.api_key = 'sk-proj-UnzdWuWBs7ZQRbRPiRCoT3BlbkFJhPM1p7DdZUMklcpnWK1S'

CORS(app, resources={r"/*": {"origins": "*"}})

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

# Fetch schema with column names and data types
def fetch_schema_with_data_types(conn):
    try:
        query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        """
        schema_df = pd.read_sql(query, conn)
        return schema_df
    except Exception as e:
        print(f"Error fetching schema with data types: {e}")
        raise

# Format schema as a string for the prompt
def format_schema(schema_df):
    schema_str = ""
    grouped = schema_df.groupby('table_name')
    for table_name, group in grouped:
        columns = ', '.join([f"{row['column_name']} ({row['data_type']})" for _, row in group.iterrows()])
        schema_str += f"{table_name}: {columns}\n"
    return schema_str
#Fetch query explainer text
def fetch_query_explaination(text):
    match = re.search(r'(.*?):', text)

    # Print the result if found
    if match:
        return (match.group(1))

# Function to generate SQL query using GPT-4o-mini
def generate_sql_query(schema_str, user_input):
    prompt = f"""
    The database contains the following schema:
    {schema_str}

    Based on this schema and the user request:
    "{user_input}"
 
    Generate an optimized SQL query that meets the user's intent.
    The query should be efficient and use the correct table and column names.
    """

    # Call GPT-4o-mini-2024-07-18 model using chat completion API
    #rephrased the prompt
    response = openai.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in generating SQL queries, ensuring the use of appropriate operators like LIKE or expressions in sql queries like '% %' for  matches if needed. Accurately map user input to the relevant tables and columns in the database based on the provided schema, using the LIKE operator for partial matches where necessary. Handle data type mismatches explicitly by casting to the appropriate type when required, ensuring correct query execution. Additionally, Manage variations in user input, such as case sensitivity or small spelling differences, using flexible matching techniques to generate precise and reliable SQL queries."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,  # Reduced token limit for completion
        temperature=0.7
    )

    # Extract SQL query from the response
    sql_response = response.choices[0].message.content
    # Find and clean the SQL query part
    start = sql_response.find("```sql") + 6
    end = sql_response.find("```", start)
    sql_query = sql_response
    

    return sql_query,sql_response

# Extract generated SQL Query
def extract_sql_query(response):
    start = response.find("```sql") + len("```sql\n")
    end = response.find("```", start)
    sql_query = response[start:end].strip()
    return sql_query

# Initialize OpenAI Chat model
openai_model = ChatOpenAI(
    openai_api_key=openai.api_key,
    model_name="gpt-4o-mini-2024-07-18",
    temperature=0.7,
    max_tokens=150
)

#Generate Response
# Update the generate_response function
def generate_response(user_query, sql_result):
    # Prepare the prompt for GPT-4 to generate the natural language response
    prompt = f"User query: \"{user_query}\"\nSQL result: {sql_result}\nGenerate a natural language response from the result:"
    
    # Call the OpenAI Chat API
    response = openai.chat.completions.create(
      model="gpt-4o-mini-2024-07-18",
      messages=[
          {"role": "user", "content": prompt}
      ],
      max_tokens=500,
      temperature=0.7
    )
    
    return response.choices[0].message.content

# Make sure to replace the completion calls elsewhere in the code

    
# Create a ChatPromptTemplate with the knowledge base included
template = """
## Knowledge Base:
{knowledge_base}

## Database Schema:
{database_schema}

## Question:
{question}

## Answer:
"""

prompt_template = ChatPromptTemplate.from_template(template)

def get_answer_from_chatbot(question, database_schema):
    try:
        prompt = prompt_template.format(
            knowledge_base="",
            database_schema=database_schema,
            question=question
        )
        response = openai_model.invoke(input=prompt)
        parsed_response = response.content.strip() if hasattr(response, 'content') else "No response content found."
        return parsed_response
    except Exception as e:
        return f"Error generating response from OpenAI: {str(e)}"
        
# Function to execute the SQL query and print the results
def execute_sql_query(conn, sql_query):
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            return results
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return None
        
# Determine if user query is related to database or general knowledge
def determine_query_type(user_query, schema_df, threshold=75):
    user_query_lower = user_query.lower()
    
    # Extract unique table and column names from the schema and convert to lowercase
    table_names = schema_df['table_name'].str.lower().unique()
    column_names = schema_df['column_name'].str.lower().unique()
    
    # Function to check fuzzy match
    def is_fuzzy_match(query, options, threshold):
        for option in options:
            if fuzz.partial_ratio(query, option) >= threshold:
                return True
        return False
    
    # Check if user query matches any table or column name
    if is_fuzzy_match(user_query_lower, table_names, threshold) or \
       is_fuzzy_match(user_query_lower, column_names, threshold):
        return "database"
    
    return "knowledge"

# Main function to handle user queries
def process_user_query(user_input):
    conn = connect_to_db()
    schema_df = fetch_schema_with_data_types(conn)
    processed_schema_str = format_schema(schema_df)
    query_type = determine_query_type(user_input, schema_df)

    if query_type == "database":
        sql_query,sql_response = generate_sql_query(processed_schema_str, user_input)
        sql_query=extract_sql_query(sql_query)
        explain_text=fetch_query_explaination(sql_response)
        print("Query Explaination:",explain_text)
        print("Generated SQL Query:", sql_query)

        
        # Execute the generated SQL query
        results = execute_sql_query(conn, sql_query)
        rows=results
        # print("Row:",rows)
        if len(rows)!=0:
            resp=generate_response(user_input,rows)
            result=resp+"\n"+explain_text
            return result
        else:
            return "I'm sorry, but I'm unable to provide results. Could you please clarify your query so I can assist you better?"
        
        conn.close()
    
    else:
        # For non-database related queries, respond using the chatbot
        return get_answer_from_chatbot(user_input, processed_schema_str)

# # Example usage
# if __name__ == "__main__":
#     while True:
#         user_input = input("Enter your query: ")
#         if user_input.lower() in ['exit', 'quit']:
#             break
#         response = process_user_query(user_input)


# This is the Python function where you'll process the received input
def process_input(result):
    return f"{result}"

# Route to receive data from another service
@app.route('/receive_data', methods=['POST'])
def receive_data():
    # Get the data sent in the POST request (assumes JSON format)
    data = request.json.get('data')
    
    # Pass the received data to your processing function
    result=process_user_query(data)
    processed_result = process_input(result)
    
    # Send the processed result back as a response
    return jsonify({'result': processed_result})

if __name__ == '__main__':
    app.run(port=5000, debug=True)  # You can change this port if needed



