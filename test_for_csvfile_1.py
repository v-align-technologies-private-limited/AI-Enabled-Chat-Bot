from initial_config import *
from langchain import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import pandas as pd
import logging
# Set up logging
logging.basicConfig(filename='agent_logs.txt', level=logging.DEBUG)

# Load a smaller dataset to reduce processing time
filepath = "./heart.csv"
data = pd.read_csv(filepath, nrows=50)  # Using 50 rows to simplify for debugging
data.to_csv("temp_heart.csv", index=False)  # Save as a temporary CSV with only 50 rows

# Initialize model and agent
p = Initialize_config()
p.process_openAI_model()
model = p.openai_model
print("Loaded OpenAI model")

# Create the agent with high iteration and time limits
agent = create_csv_agent(
    model,
    "temp_heart.csv",
    verbose=True,  # Set to True for detailed logging
    allow_dangerous_code=True,
    max_iterations=20,        # Higher iteration limit
    max_execution_time=120    # Higher time limit (2 minutes)
)

# Loop for questions
choice = 1
while choice == 1:
    prompt = input("Enter your question: ")
    try:
        final_answer = agent.run(prompt)
        print("Final Answer:", final_answer)
    except Exception as e:
        print("An error occurred:", e)
    
    choice = int(input("Do you want to continue (1/0): "))



###################################################################################33

