import pandas as pd
import re
import json
import numpy as np
from openai_manager1 import OpenAI_manager
import torch
from symentic_word_manager import *
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
        self.tokenizer = None
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

    def call_query_pinecone(self, user_input, pinecone_index, embedding_model,tokenizer):
        swm=Symentic_word_manager()
        # Set model and index
        self.pinecone_index = pinecone_index
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.augmented_input = user_input

        # Query Pinecone for each feature in namespace
        for namespace, columns in self.cleaned_feature_dict.items():
            self.augmented_input = self.query_pinecone_and_augment_input(self.augmented_input, namespace, columns,swm)
        print(self.augmented_input)
        return self.augmented_input

    def query_pinecone_and_augment_input(self, input_text, namespace, columns,swm):
        
        for column_name, entity_value in columns.items():
            if column_name in self.searched_cols or not entity_value:
                continue  # Skip if already searched or if no entity value

            self.searched_cols.add(column_name)

            try:
                # Generate query embedding
                #query_embedding = np.array(self.embedding_model.encode([entity_value])[0], dtype=np.float32)
                ev=entity_value.split()
                status=0
                if len(ev)>1:
                    entity_value_gen=swm.bigram_to_unigram(entity_value)
                    status=1
                if status==1:
                    inputs = self.tokenizer(entity_value_gen, return_tensors="pt", padding=True, truncation=True)
                else:
                    inputs = self.tokenizer(entity_value, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy().astype(np.float32)
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
                print("Enity value:",entity_value,"\nand matches for it\n")
                print(matches[0]['metadata'].get('unique_value', entity_value),"score:",matches[0].get("score"))
                print(matches[1]['metadata'].get('unique_value', entity_value),"score:",matches[1].get("score"))
                print(matches[2]['metadata'].get('unique_value', entity_value),"score:",matches[2].get("score"))
                
                if matches:
                    best_match = max(matches, key=lambda match: match['score']) if matches else None
                    # Extract the best score
                    best_score = best_match['score'] if best_match else None
                    best_match=best_match['metadata']['unique_value']
                    print("entity_value:",entity_value)
                    print("best_match:",best_match)
                    print("best_score:",best_score)
                    input_text = input_text.replace(entity_value, best_match)
                    #print(input_text)
                else:
                    print(f"No matches found for {entity_value} in Pinecone.")

            except Exception as e:
                print(f"Error querying Pinecone for {entity_value}: {str(e)} with new code")
        return input_text
