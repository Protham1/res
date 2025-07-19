import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Simple math prompt (inequality solving)
prompt = "Solve the inequality: 2x - 5 > 3. Show the steps."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=100
)

print("🧮 GPT-4o-mini Response:")
print(response.choices[0].message.content)
