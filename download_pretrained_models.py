import os
from transformers import AutoModelForMaskedLM, AutoTokenizer, Wav2Vec2FeatureExtractor, HubertModel

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

def download_hubert_model(model_name, save_directory):
    """Downloads a Hugging Face Hubert model and feature extractor to a specified directory."""
    if os.path.exists(save_directory) and os.listdir(save_directory):
        print(f"Hubert model already exists in {save_directory}. Skipping download.")
        return

    print(f"Downloading Hubert model {model_name} to {save_directory}...")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = HubertModel.from_pretrained(model_name)

        os.makedirs(save_directory, exist_ok=True)
        
        feature_extractor.save_pretrained(save_directory)
        model.save_pretrained(save_directory)
        
        print("Hubert model downloaded and saved successfully.")
    except Exception as e:
        print(f"An error occurred during Hubert model download: {e}")

if __name__ == "__main__":
    # Download BERT model
    bert_model_name = "hfl/chinese-roberta-wwm-ext-large"
    bert_save_dir = "pretrained_models/bert"
    download_model(bert_model_name, bert_save_dir)

    # Download Chinese Hubert Base model
    hubert_model_name = "TencentGameMate/chinese-hubert-base"
    hubert_save_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    download_hubert_model(hubert_model_name, hubert_save_dir)

    # TODO: User needs to manually download GPT models (e.g., s1.pth)
    # Place them in GPT_SoVITS/pretrained_models/
    print("\nIMPORTANT: Please manually download the GPT models (e.g., s1.pth) and place them in 'GPT_SoVITS/pretrained_models/'.")
    print("Refer to the official GPT-SoVITS GitHub repository for download links and instructions.")
