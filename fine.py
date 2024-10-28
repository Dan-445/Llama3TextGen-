from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Load the trained model and tokenizer
model_path = "/content/my_trained_model"  # Ensure this path matches your output_directory
merged_model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Check if a GPU is available, and set the device accordingly
device = 0 if torch.cuda.is_available() else -1  # device=0 for GPU, device=-1 for CPU
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Initialize the pipeline with the trained model, tokenizer, and GPU support
riddle_pipe = pipeline(
    "text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    device=device  # Explicitly set the device to GPU or CPU
)

# Example prompt to generate text from the trained model
prompt = "write a code to calculate area of circle if r=4"

# Run inference using the pipeline
output = riddle_pipe(prompt)

# Print the generated output
print(output[0]['generated_text'])