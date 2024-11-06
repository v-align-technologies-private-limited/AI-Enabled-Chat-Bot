import pandas as pd
from haystack.schema import Document

# Load the CSV data into a pandas DataFrame
df = pd.read_csv("heart.csv")

# Convert each row into a Document
documents = []
for _, row in df.iterrows():
    text = f"Age: {row['age']}, Sex: {row['sex']}, CP: {row['cp']}, Target: {row['target']}"
    documents.append(Document(content=text))

# You can use this `documents` list to create a DocumentStore
from haystack.document_stores import InMemoryDocumentStore

# Create a document store (in-memory)
document_store = InMemoryDocumentStore(use_bm25=True)

# Write documents to the document store
document_store.write_documents(documents)
from haystack.nodes import BM25Retriever

# Set up a BM25 retriever
retriever = BM25Retriever(document_store=document_store)
from haystack.nodes import FARMReader

# Load a pre-trained reader model (like BERT)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
from haystack.pipelines import ExtractiveQAPipeline

# Set up the Extractive QA pipeline
pipe = ExtractiveQAPipeline(reader, retriever)
# Ask a question
query = "What is the average age?"

# Run the query through the pipeline
result = pipe.run(query=query, params={"Retriever": {"top_k": 100}, "Reader": {"top_k": 10}})

# Display the result
print(result['answers'])
