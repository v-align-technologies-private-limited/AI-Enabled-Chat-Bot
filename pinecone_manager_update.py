import pandas as pd
import re
import json
import numpy as np
from openai_manager import *
from initial_config import *
from db_manager import *

class Pinecone_manager:
    def __init__(self, schema_df):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames = {}
        self.searched_cols = []
        self.searched_tables = []
        self.augmented_input = ''
        self.intermediate_input=''
        self.schema_df = schema_df
        self.extracted_Features = None
        self.cleaned_feature_dict = None
        self.tokenizer= None
        self.pinecone_index = None
        self.selection={}
        self.selection_required=False
        self.ic=Initialize_config()

    def clear_all(self):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames = {}
        self.searched_cols = []
        self.searched_tables = []
        self.augmented_input = ''
        self.intermediate_input=''
        self.selection={}
        self.selection_required=False

    def process_user_input(self, user_input):
        self.extracted_Features = OpenAI_manager.extract_features_with_openai(OpenAI_manager, user_input, self.schema_df)
        #print(self.extracted_Features)

    def process_extracted_features(self):
        def clean_extracted_features(feature_dict):
            print(feature_dict)
            # Remove any keys with None or empty values
            cleaned_feature_dict = {k: v for k, v in feature_dict.items() if v not in [None, '', [], {}, 'none', 'null', 'n/a', 'not specified']}
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
    #reframe the input with selected values
    def call_query_pinecone1(self, user_input, p_i, data):
        print("pineconedata", data)
        for x in data.keys():
            selected=str(data[x])
            user_input=user_input.replace(x,selected)
        self.augmented_input=user_input
        print("augumented_input", self.augmented_input)
        self.selection_required=False
        
    #check if any multiple values for each entity found in vectorDB
    def call_query_pinecone(self, user_input, p_i):
        res=''
        self.pinecone_index = p_i
        for key, val in self.cleaned_feature_dict.items():
            columns = list(val.keys())
            tables=list(val.values())
            print(columns)
            if self.augmented_input == '':
                res = self.query_pinecone_and_augment_input(user_input, key, columns,tables)
            else:
                res = self.query_pinecone_and_augment_input(self.augmented_input, key, columns,tables)
        print("augumentedinput",res)
        self.clear_all()
        return res

    def query_pinecone_and_augment_input(self, user_input, namespace, columns,tables):
        openai.api_key=self.ic.return_key()
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
        print(flat_entities)
        for column_name,table_name in zip(columns,tables):
            print(column_name)
            
            if column_name not in self.searched_cols or table_name not in self.searched_tables:
                self.searched_cols.append(column_name)
                self.searched_tables.append(table_name)

                # Obtain the entity value corresponding to the current column
                entity_value = self.cleaned_feature_dict[namespace].get(column_name, None)
                #print(entity_value)
                if not entity_value:
                    print(True)
                    continue  # Skip to the next column if no value is found

                # Generate the query embedding for the entity value
                response = openai.embeddings.create(
                    model="text-embedding-3-large",  # Correct embedding model
                    input=entity_value  # Input must be a list
                )
                embedding = response.data[0].embedding

                try:
                    result = self.pinecone_index.query(
                        namespace=namespace,
                        vector=embedding,
                        filter={"column_name": {"$eq": column_name}},
                        top_k=3,
                        include_values=True,
                        include_metadata=True
                    )

                    matches = result.get('matches', [])
                    #print(matches)
                    if matches:
                        get_match=[]
                        # Sort matches by score in descending order
                        matches.sort(key=lambda x: x['score'], reverse=True)

                        # Check if multiple matches have a significant score difference
                        best_match = matches[0]
                        print("match1:",matches[0]['metadata'].get('unique_value', entity_value))
                        print("match2:",matches[1]['metadata'].get('unique_value', entity_value))
                        print("match3:",matches[2]['metadata'].get('unique_value', entity_value))
                        print("Best match:",matches[0]['metadata'].get('unique_value', entity_value))
                        best_score = best_match['score']
                        print("Best Score:",best_score)
                        selection_required = False
                        selected_match = best_match['metadata'].get('unique_value', entity_value)

                        # Check if any other match has a score difference < 0.1
                        for match in matches[1:]:
                            print("MAtch score:",match['score'])
                            score_diff = best_score - match['score']
                            if score_diff < 0.07:
                                selection_required = True
                                break
                            else:
                                continue
                                
                        if selection_required:
                            # Record the values for multiple values to select among the matches
                            print(f"Multiple matches found with significant score difference for '{entity_value}'. Please select:")
                            for i, match in enumerate(matches):
                                get_match.append(match['metadata'].get('unique_value', entity_value))
                            self.selection[entity_value]=get_match
                            self.selection_required=True
                        else:
                            best_match_for_1_entity = matches[0]['metadata'].get('unique_value', entity_value)
                            #print('best_match_for_1_entity', best_match_for_1_entity)
                            self.augmented_input = self.augmented_input.replace(entity_value, best_match_for_1_entity)

                        
                    else:
                        print(f"No matches found for {entity_value} in Pinecone.")
                except Exception as e:
                    print(f"Error querying Pinecone: {str(e)}")
            else:
                print("Column already searched")
        if self.selection_required==True:
            
            print("Selection dict:",self.selection)
            print("Recent:",self.intermediate_input)
            return {"selection": self.selection}
        else:
            return self.augmented_input
