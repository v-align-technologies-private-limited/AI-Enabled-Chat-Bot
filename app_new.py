from flask import Flask, request, jsonify
from flask_cors import CORS
from db_manager import *
from initial_config1 import *
from db_manager import *
from schema_manager import *
from pinecone_manager1 import *
from openai_manager import *
from query_manager import *
from symentic_word_manager import *
import warnings
DB,p,conn,openai_manager=None,None,None,None
user_input=''
DB=DB_Manager()
openai_manager=OpenAI_manager()
p=Initialize_config()
p.assign_pinecone_index()
p.process_openAI_model()
p.set_prompt_template()
db_name="zoho_projects_data"
conn = DB.connect(DATABASE_DB = f"{db_name}")
app = Flask(__name__)
CORS(app)
schema='public'
query = f"""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
        """
#set schema
schema_manager=Schema_manager(conn,query,schema)
#Fetch the DB schema
schema_manager.fetch_schema_with_data_types()
#format the schema as required
schema_manager.format_schema()
print("Schema:",schema_manager.schema_df)
#create synthetic word instance to use parellel words
swm=Symentic_word_manager()                              
user_input=''
def main(db_name='',schema='',data='',determine_querry=''):
        determine_querry.determine_query_type(data)
        if determine_querry.query_type=="database" and data.lower()!='hi':
                pine_cone=Pinecone_manager(schema_manager.schema_df)
                pine_cone.process_user_input(user_input)
                _, feature_list=pine_cone.process_extracted_features()
                print(feature_list)
                if len(feature_list)!=0:
                        pine_cone.call_query_pinecone(user_input,p.pinecone_index,p.embedding_model,p.tokenizer,swm)
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
                    openai_manager.generate_sql_query(schema_manager.schema_str,data)
                    
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
                DB.execute_sql_query(response)
                print(DB.results)
                openai_manager.generate_response(user_input,DB.results)
                return (openai_manager.response)
        return response
                
                
                
        #DB.close_connection()
@app.route('/process', methods=['POST'])
def process_request():
    global user_input,conn,DB
    conn=DB.connect(DATABASE_DB = f"{db_name}")
    determine_querry=Determine_querry_type(schema_manager.schema_df)
    try:
        data = request.json
        key=next(iter(data.keys()))
        data=data[key]
        user_input=data
        result=main(db_name=db_name,schema='public',data=user_input,determine_querry=determine_querry)
        #print({"result": result})
        return jsonyfy({"result": result})
        DB.close_connection()
    except Exception as e:
        DB.close_connection()
        return jsonyfy({"result":f"There is an issue with query genration, query can not be executed with selected db please provide proper query{e}"})
        #print({"result": f"There is an issue with query genration, query can not be executed with selected db please provide proper query{e}"})

@app.route('/select_db', methods=['POST'])
def assign_db():
    global db_name
    data = request.json
    key=next(iter(data.keys()))
    db_name=data[key]
    return jsonify({"result":"DB selected successfully"})
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True,port=5001)

        

        
    
