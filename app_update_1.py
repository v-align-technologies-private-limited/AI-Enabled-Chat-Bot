from flask import Flask, request, jsonify
from flask_cors import CORS
from db_manager import *
from initial_config import *
from db_manager import *
from schema_manager import *
from pinecone_manager import *
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
logs={}
db_name="python_test_poc_two"
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
                                    
user_input=''
def main(db_name='',schema='',data='',determine_querry='',key=''):
        global user_input,logs
        determine_querry.determine_query_type(data)
        if determine_querry.query_type=="database":
                print(4)
                pine_cone=Pinecone_manager(schema_manager.schema_df)
                pine_cone.process_user_input(user_input)
                _, feature_list=pine_cone.process_extracted_features()
                print("Features:",feature_list)
                if len(feature_list)!=0:
                    #if input is user question find the matches for entities in pinecone
                    if key=="Query":
                            user_input=data
                            res=pine_cone.call_query_pinecone(user_input,p.pinecone_index,p.embedding_model)
                    #if input is user seletions for entity
                    else :
                            pine_cone.call_query_pinecone1(user_input,p.pinecone_index,p.embedding_model,data)
                            res=''
                    if isinstance(res, dict):
                            logs['options']=res
                            return res
                    logs['aug ip']=pine_cone.augmented_input
                    openai_manager.generate_sql_query(schema_manager.schema_str,pine_cone.augmented_input)
                    logs['sql']=openai_manager.sql_query
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
            response=openai_manager.get_answer_from_chatbot(user_input, schema_manager.schema_str,p.openai_model)
            if ("sql" in response.lower()) and (data.lower()!='hi'):
                start = response.find("```sql") + 6
                end = response.find("```", start)
                response = response[start:end].strip()
                print("SQL:",response)
                logs['aug ip']=user_input
                logs['sql']=openai_manager.sql_query
                logs['options']=None
                DB.execute_sql_query(response)
                print(DB.results)
                openai_manager.generate_response(user_input,DB.results)
                return (openai_manager.response)
        
app = Flask(__name__)
CORS(app)
@app.route('/process', methods=['POST'])
def process_request():
    global user_input,conn,logs
    determine_querry=Determine_querry_type(schema_manager.schema_df)
    print(1)
    try:
        data = request.json
        key=next(iter(data.keys()))
        data=data[key]
        print(2)
        if key=='Query':
            user_input=data
            print(3)
        result=main(db_name=db_name,schema='public',key=key,data=data,determine_querry=determine_querry)
        print(result)
        if isinstance(result, dict):
            return jsonify({"selection":result})
        res={"result": result}
        res.update(logs)
        logs.clear()
        return jsonify(res)
    except Exception as e:
        return jsonify({"result": str(e),'logs':"Error"})
@app.route('/select_db', methods=['POST'])
def assign_db():
    global db_name
    data = request.json
    key=next(iter(data.keys()))
    db_name=data[key]
    return jsonify({"result":"DB selected successfully"})
    

if __name__ == "__main__":
    app.run(debug=True,port=5001)

    
