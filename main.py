import os
import csv
import time
from mcts import mcts_search, ConversationState
from dotenv import load_dotenv
from google import genai
from generate_response.prompts import system_prompt, generate_context_prompt, generate_past_messages_prompt, generate_goal_prompt, generate_user_prompt, generate_response_prompt, generate_message_type_prompt
from evaluate import score_message

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)

user_context = "I am a 20 year old college student who is trying to get a job at Apple as a software engineer intern. I am using cold outreach to a recruiter to schedule an interview for a job at a tech company."
message_type = "Email thread"
past_messages = "Me: Subject: Software Engineer Intern Inquiry - John Doe Dear James, I am writing to express my interest in software engineering internship opportunities at Apple. As a third-year computer science student at the University of Virginia, I have been following Apple's work in the development of the Swift programming language and its integration with innovative technologies like SwiftUI with great interest. I would appreciate the opportunity to discuss how my skills and experiences align with Apple's internship needs. Would you be available for a brief conversation sometime next week? Thank you for considering my request. Sincerely, John Doe James: Subject: Re: Software Engineer Intern Inquiry - John Doe Hi John, Thank you for reaching out and for your interest in a software engineering internship at Apple. I'm glad to hear that you're following our work with Swift and SwiftUI. Your background as a computer science student at UVA sounds promising. I'd be happy to learn more about your experience and the specific areas of software engineering you're most passionate about. Could you share a bit more about the projects you've worked on, any relevant technical skills you’ve developed, and what you're hoping to gain from an internship at Apple? Looking forward to hearing more from you. Best, James"
goal = "I want to schedule an interview with the recruiter for an internship at Apple."
user_input = "I have an extensive background in machine learning and AI with my many projects such as a ML based trading bot. I am looking to gain valuable backened development experience from an internship at Apple."

test_prompt = system_prompt + generate_context_prompt(context=user_context) + generate_message_type_prompt(message_type=message_type) + generate_past_messages_prompt(past_messages=past_messages) + generate_goal_prompt(goal=goal) + generate_user_prompt(user_input=user_input)

print(test_prompt)

def is_too_similar(variant: str, original: str, threshold: float = 0.8):
    """
    Check if a variant is too similar to the original message.
    
    Args:
        variant (str): The variant message
        original (str): The original message
        threshold (float): Similarity threshold (0-1)
        
    Returns:
        bool: True if the messages are too similar
    """
    variant_lower = variant.lower()
    original_lower = original.lower()
    
    variant_words = set(variant_lower.split())
    original_words = set(original_lower.split())
    
    if not original_words:
        return False
    
    intersection = len(variant_words.intersection(original_words))
    union = len(variant_words.union(original_words))
    
    if union == 0:
        return False
    
    similarity = intersection / union
    
    contains_original = original_lower in variant_lower or variant_lower in original_lower
    
    return similarity > threshold or contains_original
    # return variant == original

def generate_variants(message: str, target_variants: int = 3):
    """
    Generate message variants that are different from the original message.
    
    Args:
        message (str): The original message to generate variants for
        max_attempts (int): Maximum number of generation attempts
        target_variants (int): Number of variants to generate (default: 3)
        
    Returns:
        list: List of unique message variants
    """
    variants = []
    
    while len(variants) < target_variants:
        try:
            generation_prompt = test_prompt + "\n\nGenerate exactly 1 completely unique and optimized message in the correct <MESSAGE_ TYPE> that is SUBSTANTIALLY DIFFERENT from the original message below. Use different wording, structure, and approach. Do not repeat or closely paraphrase the original message.\n\nOriginal message: " + message
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=generation_prompt
            )
            
            if hasattr(response, "text") and response.text.strip():
                variant = response.text.strip()
                
                if (not is_too_similar(variant, message) and 
                    not any(is_too_similar(variant, existing) for existing in variants)):
                    variants.append(variant)
                
        except Exception as e:
            time.sleep(1) # seemless workaround for gemini quota limitation
            
    return variants

initial_state = ConversationState(message=user_input)

try:
    best_message, all_messages = mcts_search(
        initial_state=initial_state,
        generate_variants_fn=generate_variants,
        evaluate_fn=score_message,
        iterations=15,
        return_all=True
    )

    print("\n<ALL MESSAGES>")
    
    best_message_obj = None
    other_messages = []
    
    for msg_obj in all_messages:
        if msg_obj['message'] == best_message:
            best_message_obj = msg_obj
        else:
            other_messages.append(msg_obj)
    
    sorted_other_messages = sorted(other_messages, 
                                  key=lambda x: x['final_score'],
                                  reverse=True)
    
    display_messages = [best_message_obj] if best_message_obj else []
    display_messages.extend(sorted_other_messages[:2 if best_message_obj else 3])
    
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

"""
Commented out code used for data gathering
"""
csv_file_path = "mcts_scores.csv"
csv_headers = ["message", "score", "MCTS score", "visits"]

file_exists = os.path.isfile(csv_file_path)
with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    
    if not file_exists:
        writer.writeheader()
    
    if best_message_obj:
        writer.writerow({
            "message": best_message_obj['message'],
            "score": round(best_message_obj['final_score'], 4),
            "MCTS score": round(best_message_obj['value'], 4),
            "visits": best_message_obj['visits']
        })