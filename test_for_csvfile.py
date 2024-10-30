import pandas as pd
from initial_config import *
from langchain import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Load the CSV data
data = pd.read_csv("./heart.csv", nrows=50)  # Load a maximum of 50 rows for performance
print("Data Loaded Successfully")

# Initialize the OpenAI language model
p = Initialize_config()
p.process_openAI_model()
assistant_model = p.openai_model  # Assistant model for data extraction
print("Loaded OpenAI assistant model")

# Create CSV agent for relevant data extraction
csv_agent = create_csv_agent(assistant_model, "./heart.csv", verbose=True, allow_dangerous_code=True)

# Function to get relevant data using the assistant model
def get_relevant_data(question):
    # Use the assistant model to get relevant data from the CSV
    relevant_data = csv_agent.run(question)
    print("Data:",relevant_data)
    return relevant_data

# Function to generate response using OpenAI language model
def generate_response(relevant_data, question):
    # Create a prompt with the relevant data
    prompt = f"Based on the following relevant data:\n{relevant_data}\n\nAnswer the question:\n{question}\n\nFinal Answer:"
    
    # Call the OpenAI language model to get a response
    response = assistant_model(prompt)
    return response.content.strip()  # Clean up the response

# User query loop
if __name__ == "__main__":
    choice = 1
    while choice == 1:
        question = input("Enter your question: ")
        
        # Step 1: Get relevant data
        relevant_data = get_relevant_data(question)
        
        # Step 2: Generate response based on the relevant data
        final_answer = generate_response(relevant_data, question)
        
        print("Final Answer:", final_answer)
        
        choice = int(input("Do you want to continue (1 for Yes, 0 for No): "))
