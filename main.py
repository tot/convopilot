import os
from dotenv import load_dotenv
from google import genai
from generate_response.prompts import system_prompt, generate_context_prompt, generate_past_messages_prompt, generate_goal_prompt, generate_user_prompt, generate_response_prompt
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

user_context = "I am a 20 year old college student who is trying to get a job at Apple as a software engineer intern. I am sending a direct message as cold outreach to a recruiter to schedule an interview for a job at a tech company."
past_messages = ""
goal = "I want to schedule an interview with the recruiter for an internship at Apple."
user_input = "hi, I would like to schedule an interview at your company. What is your availability?"

test_prompt = system_prompt + generate_context_prompt(context=user_context) + generate_past_messages_prompt(past_messages=past_messages) + generate_goal_prompt(goal=goal) + generate_user_prompt(user_input=user_input)

print(test_prompt)

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=test_prompt
)
print(response.text)