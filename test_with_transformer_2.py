import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import torch
from huggingface_hub import login
import logging

# Set your Hugging Face token directly
logging.basicConfig(level=logging.INFO)
os.environ["HUGGINGFACE_TOKEN"] = "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"  # replace with your actual token

# Alternatively, use the login function to authenticate
login("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")  # replace with your actual token

# Load CSV data
csv_path = 'heart.csv'  # path to your CSV file
data = pd.read_csv(csv_path)

# Create a structured text representation for each row
data['text'] = data.apply(lambda row: f"age: {row['age']}, sex: {row['sex']}, cp: {row['cp']}, target: {row['target']}", axis=1)
texts = data['text'].tolist()

# Load a model for text generation
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
qa_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Step 1: Create embeddings for each row
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for embeddings
embeddings = embedding_model.encode(texts, convert_to_tensor=True).cpu().detach().numpy()

# Step 2: Set up FAISS index for similarity search
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 distance
index.add(embeddings)  # Add embeddings to index

def retrieve_context(query, k=3):
    """Retrieve top-k relevant rows from the CSV based on similarity to the query."""
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()
    _, indices = index.search(query_embedding, k)
    return [texts[idx] for idx in indices[0]]

def generate_answer(query, context_texts):
    """Generate an answer based on the query and context."""
    context = "\n".join(context_texts)
    prompt = f"Given the following data:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = qa_pipeline(prompt, max_length=500, num_return_sequences=1, do_sample=True, top_k=50)
    generated_text = response[0]['generated_text']
    
    # Refine the answer to be a complete sentence
    answer_start_index = generated_text.find("Answer:") + len("Answer:")
    answer = generated_text[answer_start_index:].strip()
    
    return answer

# Main function to answer questions
def answer_question(query):
    # Retrieve context
    context_texts = retrieve_context(query, k=3)  # Top 3 relevant rows
    # Generate answer
    answer = generate_answer(query, context_texts)
    return answer

# Example usage
choice=1
while choice==1:
    query = input("Enter question:")
    answer = answer_question(query)
    print("Answer:", answer)
    choice=int(input("do you want to continue(0/1)?:"))
