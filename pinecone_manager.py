import pandas as pd
import re
import json
import numpy as np
from openai_manager import OpenAI_manager

class Pinecone_manager:
    def __init__(self, schema_df):
        self.namespace = []  # Replace with your namespace
        self.column_names = {}
        self.searched_cols = set()  # Track searched columns
        self.augmented_input = ''
        self.schema_df = schema_df
        self.extracted_features = None
        self.cleaned_feature_dict = {}
        self.embedding_model = None
        self.pinecone_index = None

    def clear_all(self):
        # Reset all key variables to initial states
        self.namespace.clear()
        self.column_names.clear()
        self.searched_cols.clear()
        self.augmented_input = ''

    def process_user_input(self, user_input):
        # Extract and clean features from user input
        self.extracted_features = OpenAI_manager.extract_features_with_openai(OpenAI_manager,user_input, self.schema_df)

    def process_extracted_features(self):
        def clean_extracted_features(feature_dict):
            # Filter out empty or irrelevant values
            return {k: v for k, v in feature_dict.items() if v and v not in ['none', 'null', 'n/a', 'not specified']}
        
        try:
            # Locate JSON structure within extracted features
            json_match = re.search(r'\{.*\}', self.extracted_features, re.DOTALL)
            if json_match:
                feature_dict = json.loads(json_match.group(0))
                self.cleaned_feature_dict = clean_extracted_features(feature_dict)
                return json.dumps(self.cleaned_feature_dict, indent=4), list(self.cleaned_feature_dict.values())
            else:
                print("No JSON structure found in extracted features.")
                return None, []
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing features: {e}")
            return None, []

    def extract_namespace(self):
        # Populate namespaces and column names
        for key, val in self.cleaned_feature_dict.items():
            self.namespace.append(key)
            self.column_names[key] = val

    def call_query_pinecone(self, user_input, pinecone_index, embedding_model):
        # Set model and index
        self.pinecone_index = pinecone_index
        self.embedding_model = embedding_model
        self.augmented_input = user_input

        # Query Pinecone for each feature in namespace
        for namespace, columns in self.cleaned_feature_dict.items():
            self.augmented_input = self.query_pinecone_and_augment_input(self.augmented_input, namespace, columns)
        return self.augmented_input

    def query_pinecone_and_augment_input(self, input_text, namespace, columns):
        for column_name, entity_value in columns.items():
            if column_name in self.searched_cols or not entity_value:
                continue  # Skip if already searched or if no entity value

            self.searched_cols.add(column_name)

            try:
                # Generate query embedding
                query_embedding = np.array(self.embedding_model.encode([entity_value])[0], dtype=np.float32)
                
                # Query Pinecone
                result = self.pinecone_index.query(
                    namespace=namespace,
                    vector=query_embedding.tolist(),
                    filter={"column_name": {"$eq": column_name}},
                    top_k=3,
                    include_values=True,
                    include_metadata=True
                )
                
                # Select the best match
                matches = result.get('matches', [])
                if matches:
                    best_match_score = matches[0].get('score', 0)
                    if best_match_score > 0.4:  # Threshold for high-confidence match
                        best_match = matches[0]['metadata'].get('unique_value', entity_value)
                        input_text = input_text.replace(entity_value, best_match)
                    else:
                        print(f"Low confidence match for {entity_value} in Pinecone with score: {best_match_score}")
                else:
                    print(f"No matches found for {entity_value} in Pinecone.")

            except Exception as e:
                print(f"Error querying Pinecone for {entity_value}: {str(e)}")

        print(input_text)
        return input_text
