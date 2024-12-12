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
        
    # Function to extract features with OpenAI based on user input and schema
    def extract_features_with_openai(self,user_input, processed_schema_df):
        schema_json = processed_schema_df.to_json(orient='records')
    
        # Define dynamic extraction keywords for aggregation and entity-specific queries
        dynamic_keywords = {
            "aggregation": ["highest", "most", "maximum", "max", "largest", "lowest", "least", "minimum", "min", "count", "sum", "average"],
            "entity_specific": ["who", "where", "which", "what", "how many", "how much"]
        }
    
        # Prompt preparation: Guide OpenAI to understand schema and extract relevant features
        prompt = f"""
    ### Database Schema Overview
    The following schema provides the columns and tables available in the database. Use this schema to interpret user input accurately:
    {schema_json}

    ### User Query
    The user has input: "{user_input}"

    ### Instructions
    1. **Understand User Intent**:
       - Carefully read the user query both left-to-right and right-to-left to identify potential entities, relationships, and operations.
       - Use a "chain of thoughts" approach to break down the query into parts and analyze the intent for each segment.
       - Identify if the query indicates aggregation (e.g., maximum, minimum, count) or specific entities (e.g., user names, project names, milestone names, status etc.).
       - Ensure that synonyms and closely related terms are mapped accurately to schema components.

    2. **Extract Relevant Entities**:
       - Match user input to text-based columns in the schema (e.g., `varchar`, `char`, `text`).
       - Handle partial matches, synonyms, and spelling variations by considering contextual clues from the schema.
       - Resolve foreign keys to their corresponding descriptive columns (e.g., `assigneeid` -> `username`,`projectid` -> `projectname`,`milestoneid` -> `milestonename`).

    3. **Handle Negations**:
       - Detect phrases involving negation, such as "not completed," "not blocked," or "not escalated."
      - Ensure the extracted entities reflect accurate negations.
        - For example:
          - Query: "How many projects are not blocked?"
            Response: {{
                "query_type": "aggregation",
                "intent_analysis": "The query asks for a count of projects that are not blocked, indicating a need to aggregate the number of projects based on their status.",
                "extracted_entities": {{
                    "projects_zoho_projects_": {{
                        "status": "blocked"
                    }}
                }}
            }}
          - Query: "How many projects are not completed?"
            Response: {{
                "query_type": "aggregation",
                "intent_analysis": "The query asks for a count of projects that are not completed, indicating a need to aggregate the number of projects based on their status.",
                "extracted_entities": {{
                    "projects_zoho_projects_": {{
                        "status": "completed"
                    }}
                }}
            }}


    4. **Classify Query Type**:
       - Determine if the query is:
         a. An **aggregation query**, requiring calculations like maximum, minimum, or count.
         b. An **entity-specific query**, focusing on particular items or details.
       - Provide reasoning for the classification in the output.

    5. **Expected Output Format**:
       - A JSON object containing:
         a. `query_type`: Either "aggregation" or "entity_extraction".
         b. `intent_analysis`: A breakdown of user intent and reasoning.
         c. `extracted_entities`: Relevant tables, columns, and matched values from the schema.

    6. **Handle Foreign Keys Dynamically**:
       - If a foreign key column exists (e.g., `assigneeid`,`projectid`,`milestoneid`) but points to a non-text field, exclude it.
       - If the foreign key references a string-based column (e.g., `username`,`projectname`,`milestonename`), include only the string-based column from the related table.

    ### Example Outputs
    For the query "Who logged the highest hours last month?":
    {{
        "query_type": "aggregation",
        "intent_analysis": "The query asks for the user who logged the most hours, requiring aggregation on the hours column.",
        "extracted_entities": {{
            "users_zoho_projects_": {{
                "username": "resolved dynamically"
            }}
        }}
    }}

    For the query "How many hours did Rishi log last month?":
    {{
        "query_type": "entity_extraction",
        "intent_analysis": "The query asks for specific details about a user (Rishi) and the hours logged.",
        "extracted_entities": {{
            "users_zoho_projects_": {{
                "username": "Rishi"
            }}
        }}
    }}

    For the query "What is the status of project Voice enabled which is assigned to Basavaraj?":
    {{
        "query_type": "entity_extraction",
        "intent_analysis": "The query asks for the status of a project assigned to a specific user.",
        "extracted_entities": {{
            "projects_zoho_projects_": {{
                "projectname": "Voice enabled",
                "status": "resolved dynamically"
            }},
            "users_zoho_projects_": {{
                "username": "Basavaraj"
            }}
        }}
    }}

    ### User Query Analysis and Output
    """
        try:
            # Use OpenAI to analyze the query dynamically
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are an expert assistant skilled in understanding and extracting entities from user queries based on a database schema."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=7000,
                temperature=0.2
            )
    
            # Extract the response content
            response_content = response.choices[0].message.content.strip()
            #print("response_content:", response_content)
    
            # Parse the JSON response
            extracted_features = json.loads(response_content)
            #print("extracted_features:", extracted_features)
            
            return extracted_features
    
        except openai.OpenAIError as e:
            print(f"Error with OpenAI API: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing OpenAI response: {e}")
            raise

    
    def generate_sql_query(self,processed_schema_str, intent_analysis='',augmented_input='',extracted_entities=''):
        prompt = f"""
    You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query and return ONLY the generated query. Use the following guidelines for SQL generation:
    ### Database Schema Overview
    The following schema provides the columns and tables available in the database:
    {processed_schema_str}

    ### User Query
    The user has input: "{augmented_input}"

    ### User Intent Analysis
    {intent_analysis}


    ### Instructions
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
    - The input may contain explicit instructions involving negation, such as "not completed," "not escalated," or "not blocked." In these cases:
    - **Always Use `NOT ILIKE` operator** when the negation applies to a specific value (e.g., `status NOT ILIKE '%completed%'`).
    - If the negation implies a flexible match (e.g., "not completed tasks"), use `NOT ILIKE` or `ILIKE` with `%` appropriately (e.g., `status NOT ILIKE '%completed%'`).
    - For negation, avoid assuming literal phrases like "not completed" exist in the database unless explicitly stated. Instead:
    - Translate phrases like "not completed" into conditions that exclude specific statuses (`NOT ILIKE`).
    - For all queries, validate column names and status values against the schema to ensure the query is contextually accurate.Ensure the final SQL query uses appropriate operators ( `NOT ILIKE`, etc.) for conditions involving negation.
    - Maintain precision in matching status descriptions, using `ILIKE` for case-insensitive comparisons and `%` for partial matches where necessary.
    - Generate SQL queries that reflect accurate intent by interpreting user input holistically, including implied conditions, such as negation or inclusion.

    Ensure the final SQL query is well-structured, efficient, and accurately reflects the user's request based on the provided input. The response should not include explanations, comments, or any other content beyond the generated SQL query.
    """

        try:
            # Call OpenAI to generate the SQL query
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are an expert in PostgreSQL query generation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=7000,
                temperature=0.2
            )

            # Extract generated query from response
            generated_query = response.choices[0].message.content.strip()
            print("Generated SQL Query:", generated_query)# Extract SQL query from the response
            #self.sql_response=sql_response
            # Find and clean the SQL query part
            start = generated_query.find("```sql") + 6
            end = generated_query.find("```", start)
            self.sql_query = generated_query[start:end].strip()
            print(self.sql_query)

        except openai.OpenAIError as e:
            print(f"Error with OpenAI API: {e}")
            raise

            
    def generate_response(self,aug_input,results):
        # Prepare the prompt for GPT-4 to generate the natural language response
        prompt = f"User query: \"{aug_input}\"\nSQL result: {results}\nGenerate a natural language response from the result:"
        
        # Call the OpenAI Chat API
        response = openai.chat.completions.create(
          model="gpt-4o-mini-2024-07-18",
          messages=[
              {"role": "user", "content": prompt}
          ],
          max_tokens=1000,
          temperature=0.7
        )
        
        self.response=response.choices[0].message.content
        print(self.response)
