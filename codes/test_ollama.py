#!/usr/bin/env python3
import ollama

# Test the Ollama connection with a simple prompt
model_name = "llama3:70b"

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
    print(f"Prompt: What is the capital of France?")
    print(f"Response: {response['response']}")
except Exception as e:
    print(f"Error: {e}")