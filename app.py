from flask import Flask, request, jsonify
from flask_cors import CORS
from db_manager import *
from initial_config import *
from db_manager import *
from schema_manager import *
from pinecone_manager_update import *
from openai_manager import *
from query_manager import *

import warnings
DB,p,conn,openai_manager=None,None,None,None
user_input=''
DB=DB_Manager()
openai_manager=OpenAI_manager()
p=Initialize_config()
p.assign_pinecone_index()
p.process_openAI_model()
p.set_prompt_template()
db_name="zoho_projects_data_v2_backup"
conn = DB.connect(DATABASE_DB = f"{db_name}")
app = Flask(__name__)
CORS(app)
schema='public'
query = f"""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
        """
schema_manager=Schema_manager(conn,query,schema)
schema_manager.fetch_schema_with_data_types()
schema_manager.format_schema()
logs={}
                                    
user_input=''
def main(db_name='',schema='',data='',determine_querry='',key=''):
        print("Database Schema", schema)
        global user_input,logs
        logs['options']=None
        determine_querry.determine_query_type(data)
        if determine_querry.query_type=="database" and user_input.lower()!='hi':
                
                pine_cone=Pinecone_manager(schema_manager.schema_df)
                pine_cone.process_user_input(user_input)
                _, feature_list=pine_cone.process_extracted_features()
                print(feature_list)
                if len(feature_list)!=0:
                    #checking whether user sent question 
                    if key=="Query":
                        user_input=data
                        res=pine_cone.call_query_pinecone(user_input,p.pinecone_index)
                    #if input is user seletions for entity
                    else :
                        print("callpinecone1")
                        pine_cone.call_query_pinecone1(user_input,p.pinecone_index,data)
                        res=''
                    if isinstance(res, dict):
                        logs['options']=res
                        print("Res:",res)
                        return res
                    openai_manager.generate_sql_query(schema_manager.schema_str,pine_cone.intent_analysis,pine_cone.augmented_input,pine_cone.pinecone_metadata_list)
                    logs['aug_ip']=pine_cone.augmented_input
                    DB.execute_sql_query(openai_manager.sql_query)
                    logs['sql']=openai_manager.sql_query
                    openai_manager.generate_response(pine_cone.augmented_input,DB.results)
                    pine_cone.clear_all()
                    return (openai_manager.response)
   
                else:
                    openai_manager.generate_sql_query(schema_manager.schema_str,data)
                    logs['options']=None
                    logs['aug_ip']=data
                    logs['sql']=openai_manager.sql_query
                    print("SQL:",openai_manager.sql_query)
                    DB.execute_sql_query(openai_manager.sql_query)
                    openai_manager.generate_response(user_input,DB.results)
                    return (openai_manager.response)
                
        else:
            response=openai_manager.get_answer_from_chatbot(user_input, schema_manager.schema_str,p.openai_model)
            if ("sql" in response.lower()) and (user_input.lower()!='hi'):
                start = response.find("```sql") + 6
                end = response.find("```", start)
                response = response[start:end].strip()
                logs['aug_ip']=user_input
                logs['sql']=openai_manager.sql_query
                print("SQL:",response)
                print("Basavaraj Not Reconizsed")
                DB.execute_sql_query(response)
                print(DB.results)               
                openai_manager.generate_response(user_input,DB.results)
                return (openai_manager.response)
            else:
                logs['aug_ip']=user_input
                logs['sql']=None
              
        return response
                
                
                
        #DB.close_connection()
@app.route('/process', methods=['POST'])
def process_request():
    global user_input,conn,logs,DB
    conn=DB.connect(DATABASE_DB = f"{db_name}")
    determine_querry=Determine_querry_type(schema_manager.schema_df)
    try:
        data = request.json
        print('data', data)
        key=next(iter(data.keys()))
        print('key',key)
        data=data[key]
        # print('data', data)
        # print(data)
        if key=='Query':
            user_input=data
            print(3)
        result=main(db_name=db_name,schema='public',key=key,data=data,determine_querry=determine_querry)
        if isinstance(result, dict):
            print("Result in process:",result)
            return jsonify(result)
        res={"result": result}
        res.update(logs)
        logs.clear()
        #DB.close_connection()
        print("Selection Results", res)   
        return jsonify(res)
    except Exception as e:
        DB.close_connection()
        return jsonify({"result": str(e),'logs':"Error"})
@app.route('/select_db', methods=['POST'])
def assign_db():
    global db_name
    data = request.json
    key=next(iter(data.keys()))
    db_name=data[key]
    return jsonify({"result":"DB selected successfully"})
if __name__ == "__main__":
    from waitress import serve
    warnings.filterwarnings("ignore")
    try:
        serve(app, host="0.0.0.0", port=5001)
    except Exception as e:
        print(f"Error starting server: {e}")


