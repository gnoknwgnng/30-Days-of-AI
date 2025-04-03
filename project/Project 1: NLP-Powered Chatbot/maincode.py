import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random

# Initialize lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Define intents: patterns (user inputs) and responses
intents = {
    "greeting": {
        "patterns": ["hi", "hello", "hey", "good morning"],
        "responses": ["Hello there!", "Hi! How can I help you?", "Hey, nice to see you!"]
    },
    "farewell": {
        "patterns": ["bye", "goodbye", "see you", "take care"],
        "responses": ["Goodbye!", "See you later!", "Take care!"]
    },
    "how_are_you": {
        "patterns": ["how are you", "how you doing", "are you okay"],
        "responses": ["I'm great, thanks! How about you?", "Doing awesome, how are you?"]
    },
    "name": {
        "patterns": ["what's your name", "who are you", "your name"],
        "responses": ["I'm Grok, your friendly AI assistant!", "Call me Grok, nice to meet you!"]
    },
    "default": {
        "patterns": [],
        "responses": ["Sorry, I didn’t catch that. Can you say it differently?"]
    }
}

def process_input(user_input):
    # Tokenize and lemmatize the user input
    tokens = word_tokenize(user_input.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def get_intent(tokens):
    # Match tokenized input to intent patterns
    for intent, data in intents.items():
        for pattern in data["patterns"]:
            pattern_tokens = word_tokenize(pattern.lower())
            pattern_tokens = [lemmatizer.lemmatize(token) for token in pattern_tokens]
            if all(token in tokens for token in pattern_tokens):
                return intent
    return "default"

def chatbot():
    print("Hello! I'm your NLP-powered chatbot. Type 'exit' to stop.")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Exit condition
        if user_input.lower() == "exit":
            print("Chatbot: Bye for now!")
            break
        
        # Process input and get intent
        tokens = process_input(user_input)
        intent = get_intent(tokens)
        
        # Select a random response from the matched intent
        response = random.choice(intents[intent]["responses"])
        print("Chatbot:", response)

# Run the chatbot
if __name__ == "__main__":
    chatbot()