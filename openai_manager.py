from initial_config import *
import pandas as pd
import re
import json
import numpy as np
from langchain.prompts import ChatPromptTemplate
import os
from datetime import datetime
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

        # Refined prompt to ensure OpenAI extracts only relevant string entities
        prompt = f"""
        ## Database Schema Context:
        The following represents the columns, their respective tables, and data types available in the database:
        {schema_json}

        ## User Input:
        The user has provided the following input: "{user_input}"

        ## Task:
        Extract relevant features, values, and table names from the user input based on the schema. Focus on extracting values from columns that have varchar, char, or text data types.

        ## Instructions:
        - Identify and return only those features which correspond to varchar, char, or text columns in the schema.
        - Ignore any references to columns with data types like datetime, date, int, or float unless they are part of the aggregation or filter criteria.
        - If the input includes aggregation keywords but also mentions specific column values, extract those entities.
        - Return a JSON dictionary that includes the table names as keys, and within each table, include the fields and their corresponding extracted values.
        - Omit any fields or tables where the value is empty or null.
        - Format the output as a JSON object with keys only for tables and fields that have values.
        """

        try:
            # Use the correct OpenAI chat completion method with the refined prompt
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in extracting entities that map user input to the relevant tables and columns in the database."}, 
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
    def generate_sql_query(self,processed_schema_str, aug_input):
        prompt = f"""
        You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query and return ONLY the generated query. Use the following guidelines for SQL generation:

        - The input may contain partial or similar names for entities, such as project names, user names, or status descriptions. Handle these by using `ILIKE` operators with `%` on both sides of the string to allow flexible matching.
        - Ensure you match project names like 'IIFL Samasta', 'IIfl Samasta', 'IIFL', 'iifl smasta', 'Iiflsmsota' (or any other spelling variations) to the correct project name from the database.
        - Use the column `instnm` whenever the question is about institute names and ensure it is associated with the column `unitid` in the query.
        - If context involves more than one table, use JOIN operations, but only join on columns that are correctly related, such as using `unitid` for table joins.
        - When calculating averages or ratios, ensure proper aggregation with the `AVG()` or relevant functions.
        - Pay close attention to the filtering criteria provided in the input and apply them in the `WHERE` clause using logical operators like `AND`, `OR` for combining conditions.
        - Use appropriate date or timestamp functions such as `TO_TIMESTAMP()` for proper date handling, and use `DATE_PART` or `EXTRACT` when necessary for extracting specific parts of a date.
        - If grouping is required (e.g., for totals or averages by categories), use the `GROUP BY` clause effectively.
        - For readability, use aliases for tables and columns, especially for complex joins or subqueries.
        - Where necessary, use subqueries or Common Table Expressions (CTEs) to break down complex queries into simpler parts for clarity.
        - If a limit on the number of rows is required, do not return more than 100 rows in the query.

        Make sure the SQL query accurately reflects the user's intent based on the input question, even if no Retrieval-Augmented Generation (RAG) is needed for certain cases.

        Database schema:
        {processed_schema_str}

        User input:
        "{aug_input}"
        """

        # Call GPT-4o-mini-2024-07-18 model using chat completion API
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in generating SQL queries that map user input to the relevant tables and columns in the database."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Token limit for generated completion
            temperature=0.2  # Slight temperature for creative output
        )

        # Extract SQL query from the response
        sql_response = response.choices[0].message.content
        self.sql_response=sql_response

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
