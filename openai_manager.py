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
        ### Database Schema Overview
        The following schema provides the columns and tables available in the database. Use this schema to interpret user input accurately:
        {schema_json}
        
        ### User Query
        The user has input: "{user_input}"
        
        ### Objectives
        1. **Extract Relevant Text-Based Features**:
        - Focus solely on identifying features and values that match text-based columns (`varchar`, `char`, or `text`) in the schema.
        - **Ignore** columns with data types other than `varchar`, `char`, or `text`.
        - **Exclude** any time references unless they directly relate to specific text-based entities that add context to the user’s query. If the query only references time without relevant entities, return an empty JSON object: `{{}}`.
        
        2. **Understand User Intent**:
        - Focus on relevant entities that align with the user’s query context. For example, if the query is about projects, include only project-related fields (e.g., `project_name`) and ignore unrelated columns.
        - Ensure the response reflects table and column names that best match the user's intended context.

        3. **Output Requirements**:
        - Return a JSON object containing only tables and columns with meaningful, non-empty values.
        - Drop any fields or tables where the value is empty, `null`, or a placeholder.
        - Format the output JSON to include only keys for tables and fields with actual values. If no relevant entities are found, return an empty JSON object `{{}}`.
        - Example: if the user requests a project name and the project has no owner, exclude the `owner` field:
            ```json
            {{
                "projects": {{
                    "project_name": "Project IIFL RPA"
                }}
            }}
            ```
        
        4. **Ensure Valid JSON**:
        - Ensure the output is valid JSON containing only tables and fields with actual values. If no relevant entities are found, return an empty JSON object `{{}}`.
        
        5. **No Placeholders or Comments**:
        - Do not include placeholders, comments, or fields without actual values. Focus only on values directly tied to the schema and user’s query.
        
        6. **Contextual Clarity**:
        - Validate that chosen table and column names fit the user's question context. When in doubt, prioritize accuracy and relevance over trying to cover all possible terms.
        """

        try:
            # Use the correct OpenAI chat completion method with the refined prompt
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in extracting entities that map user input to the relevant tables and columns in the database."}, 
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5000,
                temperature=0.2
            )
            
            # Extract the response text
            extracted_features = response.choices[0].message.content.strip()

            # Remove any markdown formatting, if present
            extracted_features = extracted_features.replace('```json', '').replace('```', '').strip()
            return extracted_features
        except openai.OpenAIError as e:
            print(f"Error with OpenAI: {e}")
            raise
    
    def generate_sql_query(self,processed_schema_str, aug_input):
        prompt = f"""
            You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query and return ONLY the generated query. Use the following guidelines for SQL generation:
        
            - The input may contain partial or similar names for entities, such as project names, user names, or status descriptions. Handle these using `ILIKE` operators with `%` on both sides of the string to allow flexible matching and accurate querying.
            - **Do not use column names directly from user input** without validation. Cross-check user terms against the provided database schema and select the most relevant and existing columns.
            - If the user input mentions a column that doesn't exist (e.g., `due_date`), **substitute it with a semantically similar, valid column** from the schema (e.g., use `end_date` if `due_date` is requested but not found).
            - Ensure you match project names like 'IIFL Samasta', 'IIfl Samasta', 'IIFL', 'iifl smasta', 'Iiflsmsota' (or any spelling variations) to the correct project name from the database.
            - Use the column `instnm` whenever the question involves institute names, and ensure it is correctly associated with `unitid` in the query.
            - When context involves multiple tables, use JOIN operations, ensuring only appropriate columns (e.g., `unitid`) are used for these joins while maintaining referential integrity.
            - **Whenever using foreign key columns**, always **select the corresponding string column from the related table** for clarity. For example, if `owner_id` from the `projects` table is a foreign key referencing `user_id` in the `users` table, select the `user_name` (or relevant name column) from the `users` table, not just the ID.
            - For calculations such as averages or ratios, ensure proper aggregation using `AVG()`, `SUM()`, or other relevant functions, maintaining clarity and correctness in grouped results.
            - Apply detailed filtering criteria from the input in the `WHERE` clause, using logical operators like `AND` and `OR` to combine conditions effectively.
            - Use date and timestamp functions like `TO_TIMESTAMP()` and `EXTRACT` to handle date parsing and extraction of date components as needed.
            - Employ the `GROUP BY` clause when aggregating by categories or attributes for clear grouping.
            - Maintain readability with table and column aliases, especially for complex joins or subqueries.
            - Utilize subqueries or Common Table Expressions (CTEs) for breaking down complex logic into simpler, modular parts for better performance and clarity.
            - Limit the number of rows in the result set to a maximum of 100 rows unless specified otherwise to optimize performance and avoid excessive data output.
        
            - Ensure comprehensive understanding of user input and intent by interpreting queries from both left to right and right to left to capture full context.
            - Use a range of SQL operators (`=`, `!=`, `LIKE`, `ILIKE`, `IN`) and functions (`IFNULL`, `ISNULL`, `COALESCE`) as needed to handle specific conditions, null handling, and flexible matching.
            - Integrate advanced SQL clauses like `CASE` and `HAVING` when addressing conditional logic or grouped results.
            - Safeguard the handling of NULLs to avoid logical errors, ensuring expressions that involve NULLs behave as expected.
            - Address complex filtering by using nested conditions and logical operators, aligning the SQL with the nuanced requirements of the user query.
            - Confirm all table relationships in JOINs are clear and maintain referential integrity to support correct data linkage.
            - Prioritize performance by avoiding unnecessary complexity or inefficient constructs, focusing on optimized query design.
            - Handle edge cases where user input may imply similar or interchangeable column names, ensuring the selected columns are contextually appropriate (e.g., using `end_date` instead of `due_date` if the former is present in the schema).
            - **For foreign key relationships, always ensure that the related table’s string column (such as user names, project names, etc.) is also selected when referring to the foreign key.**
        
            Ensure the final SQL query is well-structured, efficient, and accurately reflects the user's request based on the provided input. The response should not include explanations, comments, or any other content beyond the generated SQL query.
        
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
            max_tokens=5000,  # Token limit for generated completion
            temperature=0.2  # Slight temperature for creative output
        )

        # Extract SQL query from the response
        sql_response = response.choices[0].message.content
        print(f"sql_response: {sql_response}")
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
