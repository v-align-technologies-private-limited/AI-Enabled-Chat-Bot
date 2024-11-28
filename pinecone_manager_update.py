import pandas as pd
import re
import json
import numpy as np
from openai_manager import *
from initial_config import *

class Pinecone_manager:
    def __init__(self, schema_df):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames = {}
        self.searched_cols = []
        self.augmented_input = ''
        self.intermediate_input=''
        self.schema_df = schema_df
        self.extracted_Features = None
        self.cleaned_feature_dict = None
        self.embedding_model = None
        self.tokenizer= None
        self.pinecone_index = None
        self.selection={}
        self.selection_required=False

    def clear_all(self):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames = {}
        self.searched_cols = []
        self.augmented_input = ''
        self.intermediate_input=''
        self.selection={}
        self.selection_required=False

    def process_user_input(self, user_input):
        self.extracted_Features = OpenAI_manager.extract_features_with_openai(OpenAI_manager, user_input, self.schema_df)

    def process_extracted_features(self):
        def clean_extracted_features(feature_dict):
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
    def call_query_pinecone1(self, user_input, p_i, e_model, data):
        print(data)
        for x in data.keys():
            print(x)
            pattern = re.escape(x)
            user_input=re.sub(pattern, data[x], user_input, flags=re.IGNORECASE)
        self.augmented_input=user_input
        self.selection_required=False
        
    #check if any multiple values for each entity found in vectorDB
    def call_query_pinecone(self, user_input, p_i, e_model, tokenizer):
        self.pinecone_index = p_i
        self.embedding_model = e_model
        self.tokenizer=tokenizer
        for key, val in self.cleaned_feature_dict.items():
            columns = list(val.keys())
            if self.augmented_input == '':
                res = self.query_pinecone_and_augment_input(user_input, key, columns)
            else:
                res = self.query_pinecone_and_augment_input(self.augmented_input, key, columns)
        return res

    def query_pinecone_and_augment_input(self, user_input, namespace, columns):
        openai1=Initialize_config.return_key()
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
        for column_name in columns:
            
            if column_name not in self.searched_cols:
                self.searched_cols.append(column_name)

                # Obtain the entity value corresponding to the current column
                entity_value = self.cleaned_feature_dict[namespace].get(column_name, None)
                if not entity_value:
                    continue  # Skip to the next column if no value is found

                # Generate the query embedding for the entity value
                response = openai1.embeddings.create(
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
                    if matches:
                        get_match=[]
                        # Sort matches by score in descending order
                        matches.sort(key=lambda x: x['score'], reverse=True)

                        # Check if multiple matches have a significant score difference
                        best_match = matches[0]
                        
                        best_score = best_match['score']
                        print("Best Score:",best_score)
                        selection_required = False
                        selected_match = best_match['metadata'].get('unique_value', entity_value)

                        # Check if any other match has a score difference < 0.1
                        for match in matches[1:]:
                            print("MAtch score:",match['score'])
                            score_diff = best_score - match['score']
                            if score_diff>0.04:
                                selection_required = True
                                break
                            else:
                                best_match_for_1_entity = matches[0]['metadata'].get('unique_value', entity_value)
                                self.augmented_input = self.augmented_input.replace(entity_value, best_match_for_1_entity)

                        if selection_required:
                            # Record the values for multiple values to select among the matches
                            print(f"Multiple matches found with significant score difference for '{entity_value}'. Please select:")
                            for i, match in enumerate(matches):
                                get_match.append(match['metadata'].get('unique_value', entity_value))
                            self.selection[entity_value]=get_match
                            self.selection_required=True
                        # Automatically select the best match if no significant score difference
                        else:
                            best_match_for_1_entity = matches[0]['metadata'].get('unique_value', entity_value)
                            self.augmented_input = self.augmented_input.replace(entity_value, best_match_for_1_entity)
                        self.intermediate_input=self.augmented_input
                    else:
                        print(f"No matches found for {entity_value} in Pinecone.")
                except Exception as e:
                    print(f"Error querying Pinecone: {str(e)}")
        if self.selection_required==True:
            print("Selection dict:",self.selection)
            print("Recent:",self.intermediate_input)
            return {"selection": self.selection}
        else:
            return self.augmented_input

