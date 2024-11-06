import pandas as pd
from transformers import pipeline
from fuzzywuzzy import process  # For fuzzy matching

# Load your fine-tuned model
model_name = "fine_tuned_model"
chatbot = pipeline('text-generation', model=model_name)

# Load the dataset from Excel
def load_faqs(file_path):
    df = pd.read_excel(file_path)
    faqs = dict(zip(df["Question"], df["Answer"]))
    return faqs

faq_data = load_faqs("RSL_FAQs.xlsx")

# Maintain context using a list to hold the conversation history
conversation_history = []

# Helper function to find the closest match from FAQs
def find_closest_match(user_input):
    predefined_questions = list(faq_data.keys())
    match = process.extractOne(user_input, predefined_questions, score_cutoff=90)
    if match:
        closest_match, similarity = match
        return closest_match
    return None

def generate_response(user_input):
    # Check for input validity (empty or non-string input)
    if not isinstance(user_input, str) or not user_input.strip():
        return "Please enter a valid question."

    # Try to find the closest match using fuzzy matching
    closest_match = find_closest_match(user_input)

    # If a close match is found, return the predefined response
    if closest_match:
        return faq_data[closest_match]

    # If no close match is found, inform the user
    return "I'm sorry, I didn't understand that. Please enter a valid question."
# Test the function with various inputs
print(generate_response("How can I pay my taxes?"))  # Example question from the dataset
print(generate_response("What is the tax return policy?"))
