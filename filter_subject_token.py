import json
import spacy

# Load the English pipeline for spaCy
nlp = spacy.load("en_core_web_sm")

def extract_subject_tokens(prompt):
    # Load English tokenizer, tagger, parser, NER, and word vectors
    nlp = spacy.load("en_core_web_sm")
    
    # Process the prompt text with spaCy
    doc = nlp(prompt)
    
    # Extract subject tokens and their positions
    subject_info = {}
    for token in doc:
        # if token.pos_ in ['NOUN', 'PROPN']:  # Check if token is subject
        if token.pos_ in ['NOUN', 'PROPN']:  # Check if token is subject    
            subject_info[token.text] = token.i  # Save token text and position
    
    return subject_info


# Example usage:
# prompt = "dog and cat playing in garden"
# subject_tokens = extract_subject_tokens(prompt)
# print(subject_tokens)

# json_file = "experiments.json"

# def identify_subjects_in_prompts(json_data):
#     for category_data in json_data['experiments']:
#         print(f"Category: {category_data['category']}")
#         for subcategory_data in category_data['subcategories']:
#             print(f"Subcategory: {subcategory_data['subcategory']}")
#             for prompt in subcategory_data['prompts']:
#                 print(f"Prompt: {prompt}")
#                 subject_tokens = extract_subject_tokens(prompt)
#                 print("Subjects:", list(subject_tokens.keys()))
#                 print()

# json_data = json.load(open(json_file))
# identify_subjects_in_prompts(json_data)

