from tqdm import tqdm
import random
import math
from evaluate import score_message
from mcts import mcts_search, ConversationState
import pandas as pd

processed_data = pd.read_csv('/Users/aaron/Documents/GitHub/convopilot/processed_emails.csv')

NUM_SAMPLES = 1000  # Change this to analyze more/less

def generate_variants_fn(message):
    return [
        message[::-1],
        message.lower(),
        message.upper()
    ]

sampled_df = processed_data.sample(NUM_SAMPLES, random_state=42)
score_results = []
mcts_results = []
visit_results = []

for message in tqdm(sampled_df['processed_text'], desc="Running MCTS"):
    init_state = ConversationState(message)
    best_message, all_messages = mcts_search(
        initial_state=init_state,
        generate_variants_fn=generate_variants_fn,
        evaluate_fn=score_message,
        iterations=15,
        return_all=True
    )
    best_message_obj = None
    for msg_obj in all_messages:
        if msg_obj['message'] == best_message:
            best_message_obj = msg_obj

    score_results.append(round(best_message_obj['final_score']))
    mcts_results.append(round(best_message_obj['value']))
    visit_results.append(best_message_obj['visits'])

average_score = sum(score_results) / len(score_results)
average_mcts_score = sum(mcts_results) / len(mcts_results)
average_visits = sum(visit_results) / len(visit_results)
print(f"\nAverage score over {NUM_SAMPLES} samples: {average_score:.4f}")
print(f"\nAverage MCTS score over {NUM_SAMPLES} samples: {average_mcts_score:.4f}")
print(f"\nAverage Visits over {NUM_SAMPLES} samples: {average_visits:.4f}")