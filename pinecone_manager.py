import pandas as pd
import re
import json
import numpy as np
from openai_manager import *
class Pinecone_manager:
    def __init__(self, schema_df):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames = {}
        self.searched_cols = []
        self.augmented_input = ''
        self.schema_df = schema_df
        self.extracted_Features = None
        self.cleaned_feature_dict = None
        self.embedding_model = None
        self.pinecone_index = None

    def clear_all(self):
        self.NAMESPACE = []  # Replace with your namespace
        self.columnnames = {}
        self.searched_cols = []
        self.augmented_input = ''

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

    def call_query_pinecone(self, user_input, p_i, e_model):
        self.pinecone_index = p_i
        self.embedding_model = e_model
        for key, val in self.cleaned_feature_dict.items():
            columns = list(val.keys())
            if self.augmented_input == '':
                self.augmented_input = self.query_pinecone_and_augment_input(user_input, key, columns)
            else:
                self.augmented_input = self.query_pinecone_and_augment_input(self.augmented_input, key, columns)
        return self.augmented_input

    def query_pinecone_and_augment_input(self, user_input, namespace, columns):
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
                query_embedding = self.embedding_model.encode([entity_value])[0]
                query_embedding = np.array(query_embedding, dtype=np.float32)

                try:
                    result = self.pinecone_index.query(
                        namespace=namespace,
                        vector=query_embedding.tolist(),
                        filter={"column_name": {"$eq": column_name}},
                        top_k=3,
                        include_values=True,
                        include_metadata=True
                    )

                    matches = result.get('matches', [])
                    if matches:
                        # Automatically select the best match (first match)
                        best_match = matches[0]['metadata'].get('unique_value', entity_value)
                        self.augmented_input = self.augmented_input.replace(entity_value, best_match)
                    else:
                        print(f"No matches found for {entity_value} in Pinecone.")
                except Exception as e:
                    print(f"Error querying Pinecone: {str(e)}")

        return self.augmented_input

