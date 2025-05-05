import os
from mcts import mcts_search, ConversationState
from dotenv import load_dotenv
from google import genai
from generate_response.prompts import system_prompt, generate_context_prompt, generate_past_messages_prompt, generate_goal_prompt, generate_user_prompt, generate_response_prompt, generate_message_type_prompt
from evaluate import score_message_only

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)

user_context = "I am a 20 year old college student who is trying to get a job at Apple as a software engineer intern. I am using cold outreach to a recruiter to schedule an interview for a job at a tech company."
message_type = "DM on linkedin"
past_messages = ""
goal = "I want to schedule an interview with the recruiter for an internship at Apple."
user_input = "hi, I would like to schedule an interview at your company. What is your availability?"

test_prompt = system_prompt + generate_context_prompt(context=user_context) + generate_message_type_prompt(message_type=message_type) + generate_past_messages_prompt(past_messages=past_messages) + generate_goal_prompt(goal=goal) + generate_user_prompt(user_input=user_input)

print(test_prompt)

def generate_variants(message: str):
    """
    Generate exactly 1 message.
    
    Args:
        message (str): The message to generate variants for
        
    Returns:
        list: List of exactly 1 message variant1
    """
    prompt = "Generate exactly 1 optimized message based on these requirements: " + test_prompt
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        if hasattr(response, "text"):
            variants = [msg.strip() for msg in response.text.split("---") if msg.strip()]
            if len(variants) > 3:
                variants = variants[:3]
            elif len(variants) < 3:
                while len(variants) < 3:
                    variants.append(f"\n{message}")
            
            print(f"Generating variants...")
            return variants
        else:
            print("No text in response")
    except Exception as e:
        print(f"Error generating variants: {e}")

initial_state = ConversationState(message=user_input)

try:
    print("Starting MCTS search...")
    best_message, all_messages = mcts_search(
        initial_state=initial_state,
        generate_variants_fn=generate_variants,
        evaluate_fn=score_message_only,
        iterations=15,
        return_all=True
    )

    print("\n<ALL MESSAGES>")
    sorted_messages = sorted(all_messages, 
                            key=lambda x: (x['message'] == best_message, x['final_score']),
                            reverse=True)
    
    display_messages = sorted_messages[:3]
    
    for i, message_obj in enumerate(display_messages, 1):
        is_best = message_obj['message'] == best_message
        tag = f"MESSAGE {i} {'(BEST)' if is_best else ''}:"
        print(f"\n{tag}")
        print(message_obj['message'])
        print(f"Score: {round(message_obj['final_score'], 2)} "
              f"(Polarity: {round(message_obj['polarity'], 4)}, "
              f"Subjectivity: {round(message_obj['subjectivity'], 4)})")
        if 'visits' in message_obj:
            print(f"Visits: {message_obj['visits']}, Value: {round(message_obj['value'], 2)}")
    
    print("\n</ALL MESSAGES>")

    print("\n<MCTS BEST MESSAGE>")
    print(best_message)
    print("\n</MCTS BEST MESSAGE>")
except Exception as e:
    print(f"Error during MCTS search: {e}")
    import traceback
    traceback.print_exc()