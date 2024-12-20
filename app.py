from flask import Flask, request, jsonify, g
from flask_cors import CORS
from db_manager import *
from initial_config import *
from schema_manager import *
from pinecone_manager_update import *
from openai_manager import *
from query_manager import *
import warnings

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# App configuration
app.config['DATABASE_NAME'] = "zoho_projects_data_v2_backup"
app.config['SCHEMA'] = 'public'

@app.before_request
def setup():
    """
    Runs before each request to set up a fresh context and connections.
    """
    g.DB = DB_Manager()  # Instantiate database manager for this request
    g.openai_manager = OpenAI_manager()  # OpenAI Manager for each request
    g.pinecone_manager = Initialize_config()  # Initialize Pinecone configuration
    g.pinecone_manager.assign_pinecone_index()
    g.pinecone_manager.process_openAI_model()
    g.pinecone_manager.set_prompt_template()
    
    # Set up database schema for this request
    g.conn = g.DB.connect(DATABASE_DB=app.config['DATABASE_NAME'])
    query = f"""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{app.config['SCHEMA']}'
    """
    g.schema_manager = Schema_manager(g.conn, query, app.config['SCHEMA'])
    g.schema_manager.fetch_schema_with_data_types()
    g.schema_manager.format_schema()

@app.teardown_request
def teardown(exception):
    """
    Closes database connections after each request.
    """
    if hasattr(g, 'DB'):
        g.DB.close_connection()

def main(db_name='', schema='', data='', determine_querry='', key=''):
    """
    Core logic for handling user input and generating responses or SQL queries.
    """
    print("Database Schema", schema)
    logs = {'options': None}
    
    determine_querry.determine_query_type(data)
    if determine_querry.query_type == "database" and g.user_input.lower() != 'hi':
        pine_cone = Pinecone_manager(g.schema_manager.schema_df)
        pine_cone.process_user_input(g.user_input)
        _, feature_list = pine_cone.process_extracted_features()
        print(feature_list)
        if len(feature_list) != 0:
            if key == "Query":
                g.user_input = data
                res = pine_cone.call_query_pinecone(g.user_input, g.pinecone_manager.pinecone_index)
            else:
                pine_cone.call_query_pinecone1(g.user_input, g.pinecone_manager.pinecone_index, data)
                res = ''
            if isinstance(res, dict):
                logs['options'] = res
                return res
            g.openai_manager.generate_sql_query(g.schema_manager.schema_str, pine_cone.intent_analysis, pine_cone.augmented_input, pine_cone.pinecone_metadata_list)
            logs['aug_ip'] = pine_cone.augmented_input
            logs['sql'] = g.openai_manager.sql_query
            g.DB.execute_sql_query(g.openai_manager.sql_query)
            g.openai_manager.generate_response(pine_cone.augmented_input, g.DB.results)
            pine_cone.clear_all()
            return g.openai_manager.response
        else:
            g.openai_manager.generate_sql_query(g.schema_manager.schema_str, data)
            logs['options'] = None
            logs['aug_ip'] = data
            logs['sql'] = g.openai_manager.sql_query
            g.DB.execute_sql_query(g.openai_manager.sql_query)
            g.openai_manager.generate_response(g.user_input, g.DB.results)
            return g.openai_manager.response
    else:
        response = g.openai_manager.get_answer_from_chatbot(g.user_input, g.schema_manager.schema_str, g.pinecone_manager.openai_model)
        if "sql" in response.lower() and g.user_input.lower() != 'hi':
            start = response.find("```sql") + 6
            end = response.find("```", start)
            response = response[start:end].strip()
            logs['aug_ip'] = g.user_input
            logs['sql'] = g.openai_manager.sql_query
            g.DB.execute_sql_query(response)
            g.openai_manager.generate_response(g.user_input, g.DB.results)
            return g.openai_manager.response
        else:
            logs['aug_ip'] = g.user_input
            logs['sql'] = None
        return response

@app.route('/process', methods=['POST'])
def process_request():
    """
    Handles user requests by calling the main function.
    """
    try:
        data = request.json
        key = next(iter(data.keys()))
        data = data[key]
        if key == 'Query':
            g.user_input = data
        determine_querry = Determine_querry_type(g.schema_manager.schema_df)
        result = main(db_name=app.config['DATABASE_NAME'], schema=app.config['SCHEMA'], key=key, data=data, determine_querry=determine_querry)
        if isinstance(result, dict):
            return jsonify(result)
        logs = {"result": result}
        return jsonify(logs)
    except Exception as e:
        return jsonify({"result": str(e), "logs": "Error"})

@app.route('/select_db', methods=['POST'])
def assign_db():
    """
    Allows dynamic database selection.
    """
    try:
        data = request.json
        db_name = next(iter(data.values()))
        app.config['DATABASE_NAME'] = db_name
        return jsonify({"result": "DB selected successfully"})
    except Exception as e:
        return jsonify({"result": str(e), "logs": "Error"})

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5001)
