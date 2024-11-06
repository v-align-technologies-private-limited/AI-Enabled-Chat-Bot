import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import torch
from huggingface_hub import login
# Set your Hugging Face token directly
import logging
logging.basicConfig(level=logging.INFO)
os.environ["HUGGINGFACE_TOKEN"] = "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"  # replace with your actual token

# Alternatively, use the login function to authenticate
login("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")  # replace with your actual token

# Load CSV data
csv_path = 'heart.csv'  # path to your CSV file
data = pd.read_csv(csv_path)

# Concatenate relevant columns into a single text representation for each row
# Adjust columns to match those in your CSV
data['text'] = data.apply(lambda row: f"age: {row['age']}, sex: {row['sex']}, cp: {row['cp']}, target: {row['target']}", axis=1)
texts = data['text'].tolist()

# Step 3: Load LLaMA-2 model
# Substitute with Hugging Face model path for LLaMA-2
'''from transformers import AutoTokenizer, AutoModel
try:
    model = AutoModel.from_pretrained(model_name)
except Exception as e:
    print(e)'''
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="auto", torch_dtype=torch.float32)
# Step 1: Create embeddings for each row
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for embeddings
embeddings = embedding_model.encode(texts, convert_to_tensor=True).cpu().detach().numpy()

# Step 2: Set up FAISS index for similarity search
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 distance
index.add(embeddings)  # Add embeddings to index
qa_pipeline = pipeline('text-generation', model="meta-llama/Llama-2-7b-hf", tokenizer=tokenizer)

def retrieve_context(query, k=3):
    """Retrieve top-k relevant rows from the CSV based on similarity to the query."""
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()
    _, indices = index.search(query_embedding, k)
    return [texts[idx] for idx in indices[0]]

def generate_answer(query, context_texts):
    """Generate an answer using LLaMA-2 based on the query and context."""
    context = "\n".join(context_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = qa_pipeline(prompt, max_length=200, do_sample=True, top_k=50, num_return_sequences=1)
    return response[0]['generated_text']

# Main function to answer questions
def answer_question(query):
    # Retrieve context
    context_texts = retrieve_context(query, k=3)  # Top 3 relevant rows
    # Generate answer
    answer = generate_answer(query, context_texts)
    return answer

# Example usage
query = "What is the target for a 50-year-old male with cp level 2?"
answer = answer_question(query)
print("Answer:", answer)
