# GPT_SoVITS/config.py

def change_choices():
    print("change_choices called (placeholder)")
    return [], []

def get_weights_names():
    print("get_weights_names called (placeholder)")
    # Dummy names for now, as actual GPT models are missing
    gpt_names = ["dummy_gpt_model"]
    sovits_names = ["dummy_sovits_model"]
    return sovits_names, gpt_names

name2gpt_path = {
    "dummy_gpt_model": "GPT_SoVITS/pretrained_models/s2Gv4.pth" # Using a SoVITS model as placeholder for GPT
}
name2sovits_path = {
    "dummy_sovits_model": "GPT_SoVITS/pretrained_models/vocoder.pth"
}

pretrained_sovits_name = {
    "v3": "GPT_SoVITS/pretrained_models/vocoder.pth",
    "v4": "GPT_SoVITS/pretrained_models/s2Gv4.pth",
}
