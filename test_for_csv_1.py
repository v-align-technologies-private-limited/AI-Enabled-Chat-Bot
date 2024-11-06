import pandas as pd
from gpt4all import GPT4All
import sys

# Function to load the CSV file and preprocess data
def load_csv(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path,usecols=["age", "sex", "cp", "target"])
    return df

# Function to process the question and get an answer from GPT4All
def ask_question(model, question, csv_data):
    # Create a context for the model by converting the CSV data into a textual format
    csv_text = csv_data.to_string(index=False)  # Convert the CSV to a string (without row indices)
    
    # Create a prompt that includes the question and the CSV data context
    prompt = f"""
    The following is a table of data from a CSV file:
    {csv_text}
    
    Please answer the following question based on this data:
    {question}
    """
    
    # Get the model's response based on the context
    response = model.generate(prompt)
    return response

# Main function to interact with the user
def main():
    # Load the GPT4All model
    try:
        model = GPT4All(model_name="Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    except Exception as e:
        print(e)
        sys.exit()
            

    # Ask user to upload CSV file
    file_path = './heart.csv'

    # Load and preprocess the CSV
    csv_data = load_csv(file_path)

    # Start a loop to ask questions
    while True:
        # Ask the user for a question
        question = input("Enter your question (or type 'exit' to quit): ")

        # Exit the loop if user types 'exit'
        if question.lower() == 'exit':
            print("Exiting the Q&A system.")
            break
        
        # Get the answer from the model
        answer = ask_question(model, question, csv_data)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
