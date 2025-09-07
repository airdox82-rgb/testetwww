
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer

def download_model(model_name, save_directory):
    """Downloads a Hugging Face model and tokenizer to a specified directory."""
    if os.path.exists(save_directory) and os.listdir(save_directory):
        print(f"Model already exists in {save_directory}. Skipping download.")
        return

    print(f"Downloading model {model_name} to {save_directory}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        os.makedirs(save_directory, exist_ok=True)
        
        tokenizer.save_pretrained(save_directory)
        model.save_pretrained(save_directory)
        
        print("Model downloaded and saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # The specific BERT model required by the data preparation script
    model_name = "hfl/chinese-roberta-wwm-ext-large"
    
    # The target directory where the script expects the model
    save_dir = "pretrained_models/bert"
    
    download_model(model_name, save_dir)
