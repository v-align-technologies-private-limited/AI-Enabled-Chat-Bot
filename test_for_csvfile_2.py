import pandas as pd
from initial_config import *  # Assuming this is where your OpenAI model is initialized

# 1. Load the CSV data
filepath = "./heart.csv"  # Path to your CSV file
data = pd.read_csv(filepath)

# 2. Generate a summary or key points of the data (optional)
# You can use this if you want to summarize before sending to OpenAI
summary_data = data.describe(include='all')  # Descriptive statistics for insights

# Convert summary to string format
summary_str = summary_data.to_string()

# 3. Initialize OpenAI model
p = Initialize_config()
p.process_openAI_model()
model = p.openai_model
print("Loaded OpenAI model")

# 4. Create a prompt for insights
prompt = f"""
You are an AI assistant that generates key insights from data. Here is the summary of the data:

{summary_str}

Please provide key insights and interpretations from the data above.
"""

# 5. Generate insights using OpenAI
response = model(prompt)
print(type(response))

print("Key Insights:", response.content.strip())
print(type(response))
