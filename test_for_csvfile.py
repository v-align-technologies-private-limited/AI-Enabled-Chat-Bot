import pandas as pd
from initial_config import *
from langchain import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Load the CSV data and preprocess it
def load_and_preprocess_data(file_path):
    # Read specific columns
    data = pd.read_csv(file_path, usecols=["age", "sex", "cp", "target"])  # Load only 50 rows for performance

    # Convert categorical columns to human-readable format
    data['sex'] = data['sex'].apply(lambda x: 'Male' if x == 1 else 'Female')
    cp_mapping = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
    data['cp'] = data['cp'].map(cp_mapping)

    print("Data Loaded and Preprocessed Successfully")
    print(data.info())
    print("Sample Data:\n", data.head())
    print("Number of patients with heart disease:", data[data['target'] == 1].shape[0])

    return data

# Initialize the OpenAI language model
p = Initialize_config()
p.process_openAI_model()
assistant_model = p.openai_model  # Assistant model for data extraction
print("Loaded OpenAI assistant model")

# Load and preprocess the CSV data
data = load_and_preprocess_data("./heart.csv")

# Create CSV agent for relevant data extraction
csv_agent = create_csv_agent(
    assistant_model,
    "heart.csv",  # Pass the preprocessed DataFrame instead of the file path
    verbose=False,
    allow_dangerous_code=True,
    max_execution_time=120,
    max_iterations=110
)

# Function to get relevant data using the assistant model
def get_relevant_data(question):
    try:
        # Use 'invoke' with a dictionary for the input question
        inputs = {"input": question}  # Ensure this key matches what csv_agent expects
        relevant_data = csv_agent.invoke(inputs)
        
        # Check if relevant_data is valid
        if not relevant_data or "invalid" in str(relevant_data.values()).lower():
            raise ValueError("Model returned invalid content.")

        print("Relevant Data:", relevant_data)
        return relevant_data
    
    except ValueError as e:
        print(f"Error encountered: {e}")
        print("Retrying with modified query...")
        
        # Retry with a simpler question or fallback response
        fallback_question = "Please retrieve data about heart disease based on age, sex, chest pain, and target."
        try:
            inputs["input"] = fallback_question
            relevant_data = csv_agent.invoke(inputs)
            print("Fallback Relevant Data:", relevant_data)
            return relevant_data
        except Exception as fallback_error:
            print(f"Fallback failed: {fallback_error}")
            return "Error: Unable to retrieve relevant data"

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
