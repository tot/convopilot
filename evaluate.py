from textblob import TextBlob

def evaluate_multiple_responses(responses_text):
    """
    Parses and evaluates full message blocks, split by '---'.
    
    Args:
        responses_text (str): The full output string from the language model.
        
    Returns:
        List[dict]: Evaluations per complete message.
    """
    messages = [msg.strip() for msg in responses_text.strip().split('---') if msg.strip()]
    
    results = []
    for idx, msg in enumerate(messages, 1):
        result = score_message(message=msg)
        result["message"] = msg
        result["message_id"] = idx
        results.append(result)
    
    return results

def score_message(message: str):
    """
    Score a message using TextBlob sentiment analysis.
    
    Args:
        message (str): The message to evaluate
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    blob = TextBlob(message)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    polarity_score = max(0, 1 - abs(polarity - 0.2))       # Closer to 0.2 is ideal
    subjectivity_score = max(0, 1 - abs(subjectivity - 0.3)) # Closer to 0.3 is ideal
    
    final_score = (0.6 * polarity_score + 0.4 * subjectivity_score) * 100
    
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "final_score": final_score
    }