from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import torch

# Set Hugging Face token for optional access if required
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Define paths to the custom-trained models
model_paths = {
    "1": "/content/my_trained_model",
    "2": "/content/finetuned"
}

# Prompt the user to select the model
print("Select the model to use:")
print("1 - My Trained Model (/content/my_trained_model)")
print("2 - Finetuned Model (/content/finetuned)")
user_choice = input("Enter the number corresponding to your choice: ")

# Validate input and choose the appropriate path
if user_choice not in model_paths:
    print("Invalid choice. Please select a valid option.")
else:
    chosen_path = model_paths[user_choice]
    print(f"Loading model from: {chosen_path}")

    # Load the model and tokenizer based on user's choice
    try:
        merged_model = AutoModelForCausalLM.from_pretrained(chosen_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(chosen_path, local_files_only=True)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print("Failed to load the model:", e)

    # Set the device based on GPU availability
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # Create the pipeline with the loaded model and tokenizer
    riddle_pipe = pipeline(
        "text-generation",
        model=merged_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        device=device
    )

    # Define a sample prompt
    prompt = "Create an array of length 5 which contains all even numbers between 1 and 10. Provide a detailed explanation and include the code implementation."

    # Generate and print the output
    output = riddle_pipe(prompt)
    print(output[0]['generated_text'])
