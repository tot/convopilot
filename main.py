import os
from dotenv import load_dotenv
from google import genai
from generate_response.prompts import system_prompt, generate_context_prompt, generate_past_messages_prompt, generate_goal_prompt, generate_user_prompt, generate_response_prompt, generate_message_type_prompt
from evaluate import evaluate_multiple_responses
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

user_context = "I am a 20 year old college student who is trying to get a job at Apple as a software engineer intern. I am using cold outreach to a recruiter to schedule an interview for a job at a tech company."
message_type = "DM on linkedin"
past_messages = ""
goal = "I want to schedule an interview with the recruiter for an internship at Apple."
user_input = "hi, I would like to schedule an interview at your company. What is your availability?"

test_prompt = system_prompt + generate_context_prompt(context=user_context) + generate_message_type_prompt(message_type=message_type) +generate_past_messages_prompt(past_messages=past_messages) + generate_goal_prompt(goal=goal) + generate_user_prompt(user_input=user_input)

print(test_prompt)

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Generate 3 variations of a potential message follwing these requirements exactly: " + test_prompt
)
# print(response.text)
generated_text = response.text

evaluated = evaluate_multiple_responses(generated_text)
for result in evaluated:
    print(f"\nMessage {result['message_id']}:")
    print(result['message'])
    print(f"Score: {result['final_score']} "
          f"(Polarity: {round(result['polarity'], 4)}, "
          f"Subjectivity: {round(result['subjectivity'], 4)})")