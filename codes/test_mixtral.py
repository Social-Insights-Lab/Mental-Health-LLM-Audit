#!/usr/bin/env python3
import ollama

# Test the Mixtral model with a simple prompt
model_name = "mixtral:8x22b"

try:
    response = ollama.generate(
        model=model_name,
        prompt="What is the capital of France?",
        options={
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 50,
        }
    )
    print("Test successful!")
    print(f"Model: {model_name}")
    print(f"Prompt: What is the capital of France?")
    print(f"Response: {response['response']}")
except Exception as e:
    print(f"Error: {e}")