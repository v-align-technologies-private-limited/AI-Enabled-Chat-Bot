import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# Load CSV data
csv_path = 'heart.csv'
data = pd.read_csv(csv_path)

# Concatenate relevant columns into a single text representation for each row
data['text'] = data.apply(lambda row: f"age: {row['age']}, sex: {row['sex']}, cp: {row['cp']}, trestbps:{row['trestbps']},target: {row['target']}", axis=1)
texts = data['text'].tolist()

# Initialize Q&A model and pipeline
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Embedding model for similarity search
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(texts, convert_to_tensor=True).cpu().detach().numpy()

# Set up FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

def retrieve_context(query, k=3):
    """Retrieve top-k relevant rows from the CSV based on similarity to the query."""
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()
    _, indices = index.search(query_embedding, k)
    return [texts[idx] for idx in indices[0]]

def generate_answer(query, context_texts):
    """Generate an answer using the Q&A model based on the query and multiple contexts."""
    best_answer = None
    highest_score = 0
    
    # Retrieve answer from each context text individually
    for context in context_texts:
        response = qa_pipeline(question=query, context=context)
        if response['score'] > highest_score:
            highest_score = response['score']
            best_answer = response['answer']
    
    return best_answer

# Main function to answer questions
def answer_question(query):
    context_texts = retrieve_context(query, k=3)  # Retrieve top 3 relevant contexts
    answer = generate_answer(query, context_texts)
    return answer

# Example usage
choice=1
while choice==1:
    query = input("Enter question:")
    answer = answer_question(query)
    print("Answer:", answer)
    choice=int(input("do you want to continue(0/1)?:"))
