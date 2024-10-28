
# Llama3TextGen

A customizable text generation pipeline leveraging LLaMA 3 models, with an option to select between multiple custom-trained models for local use. This project is designed to run efficiently on local systems while providing the flexibility to switch models based on user needs.

## Features

- **Custom Model Loading**: Load and utilize your fine-tuned LLaMA 3 models locally without accessing Hugging Face Hub.
- **Model Selection**: Prompt-based selection between multiple custom-trained models to suit various text generation tasks.
- **GPU Support**: Optimized for GPU acceleration to handle complex text generation efficiently.
- **Flexible Text Generation**: Generate text for diverse use cases by simply modifying the prompt.

## Setup

### Prerequisites
- Python 3.7+
- [Transformers Library](https://huggingface.co/transformers) by Hugging Face
- PyTorch with GPU support (if available)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Dan-445/Llama3TextGen-.git
   cd Llama3TextGen
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Prepare Models**:
   - Ensure your custom-trained models are saved in directories such as `/content/my_trained_model` and `/content/finetuned`.
   - Adjust the paths as needed within the code.

## Usage

1. **Run the Notebook**:
   - Open `Kay_Jay.ipynb` in Jupyter or Google Colab.
   - Follow the prompts to select the model you wish to use (e.g., "1" for `my_trained_model` or "2" for `finetuned`).

2. **Text Generation**:
   - Modify the prompt in the code cell to generate text for various use cases.
   - Run the text generation pipeline and view the output.

### Example Prompt
```python
prompt = "Create an array of length 5 which contains all even numbers between 1 and 10. Provide a detailed explanation and include the code implementation."
```

## Project Structure

- **Kay_Jay.ipynb**: Jupyter notebook containing code for model loading, selection, and text generation.
- **requirements.txt**: List of required libraries and dependencies.
- **README.md**: Project documentation.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Please ensure compatibility with Python 3.7+ and relevant libraries.

## License

This project is licensed under the MIT License. See `LICENSE` for more information.
