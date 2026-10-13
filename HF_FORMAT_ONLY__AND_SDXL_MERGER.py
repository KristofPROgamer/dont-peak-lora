import os
import json
import torch
from safetensors.torch import load_file, save_file

# ==========================================
# 1. VALIDATION LOGIC
# ==========================================
def validate_adapter_config(config_path, name):
    print(f"\n[VALIDATING CONFIG] {name}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"CRITICAL: Cannot find config at {config_path}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    peft_type = config.get("peft_type", "UNKNOWN")
    rank = config.get("r", "UNKNOWN")
    alpha = config.get("lora_alpha", "UNKNOWN")
    targets = config.get("target_modules", "UNKNOWN")
    
    print(f"  > PEFT Type : {peft_type}")
    print(f"  > Rank (r)  : {rank}")
    print(f"  > Alpha     : {alpha}")
    
    if peft_type != "LORA":
        print(f"  ! WARNING: Expected PEFT type 'LORA', got '{peft_type}'")
    if rank == "UNKNOWN":
        print("  ! WARNING: Rank not found in config. Is this a valid PEFT config?")

    return rank, alpha

# ==========================================
# 2. FORMAT CONVERTERS
# ==========================================
def convert_to_comfy(peft_state_dict, prefix):
    """Translates HF PEFT keys into flattened ComfyUI keys with up/down/alpha syntax."""
    comfy_dict = {}
    for key, tensor in peft_state_dict.items():
        # 1. Strip the Hugging Face PEFT wrapper text
        clean_key = key.replace("base_model.model.", "")
        
        # 2. FIX: Inject "text_model." if PEFT dropped it during training
        if not clean_key.startswith("text_model.") and clean_key.startswith("encoder."):
            clean_key = "text_model." + clean_key
            
        # 3. Translate A/B to down/up and flatten dots
        if "lora_A" in clean_key:
            base_name = clean_key.replace(".lora_A.weight", "")
            comfy_key = prefix + base_name.replace(".", "_") + ".lora_down.weight"
            comfy_dict[comfy_key] = tensor
            
            # Inject Alpha tensor
            alpha_key = prefix + base_name.replace(".", "_") + ".alpha"
            comfy_dict[alpha_key] = torch.tensor(tensor.shape[0], dtype=tensor.dtype)
            
        elif "lora_B" in clean_key:
            base_name = clean_key.replace(".lora_B.weight", "")
            comfy_key = prefix + base_name.replace(".", "_") + ".lora_up.weight"
            comfy_dict[comfy_key] = tensor
            
    return comfy_dict

def convert_to_hf_sdxl(peft_state_dict, prefix):
    """Prepends diffusers namespace (text_encoder. / text_encoder_2.) to PEFT keys."""
    hf_dict = {}
    for key, tensor in peft_state_dict.items():
        # Example output: text_encoder.base_model.model.encoder.layers...
        hf_dict[prefix + key] = tensor
    return hf_dict

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
# --- Set your directories here ---
DIR_TE1 = "C:\\Users\\YOUR_USER\\Desktop\\Dont_Peak_Company_LoRA\\sd1.5\\checkpoints_phase2\\lora_epoch8_cos0.8455"
DIR_TE2 = "C:\\Users\\YOUR_USER\\Desktop\\Dont_Peak_Company_LoRA\\sdxl\\Final_phase2"

print("Starting SDXL Cipher LoRA Merger Pipeline...")

# Validate Configurations
rank_te1, alpha_te1 = validate_adapter_config(os.path.join(DIR_TE1, "adapter_config.json"), "TE1 (ViT-L)")
rank_te2, alpha_te2 = validate_adapter_config(os.path.join(DIR_TE2, "adapter_config.json"), "TE2 (ViT-bigG)")

if rank_te1 != rank_te2:
    print(f"\n[!] NOTICE: TE1 Rank ({rank_te1}) and TE2 Rank ({rank_te2}) differ. This is allowed, but uncommon.")

# Load Weights
print("\n[LOADING WEIGHTS]")
te1_state = load_file(os.path.join(DIR_TE1, "adapter_model.safetensors"))
te2_state = load_file(os.path.join(DIR_TE2, "adapter_model.safetensors"))
print(f"  > TE1 loaded: {len(te1_state)} tensors.")
print(f"  > TE2 loaded: {len(te2_state)} tensors.")

# --- Process ComfyUI Format ---
print("\n[PROCESSING COMFYUI FORMAT]")
comfy_lora = {}
te1_comfy = convert_to_comfy(te1_state, "lora_te1_")
te2_comfy = convert_to_comfy(te2_state, "lora_te2_")
comfy_lora.update(te1_comfy)
comfy_lora.update(te2_comfy)

comfy_filename = "sdxl_cipher_comfyui.safetensors"
save_file(comfy_lora, comfy_filename)
print(f"  > Saved: {comfy_filename} ({len(comfy_lora)} keys)")

# --- Process Hugging Face Diffusers Format ---
print("\n[PROCESSING HUGGING FACE FORMAT]")
hf_lora = {}
# diffusers uses 'text_encoder' for TE1 and 'text_encoder_2' for TE2
te1_hf = convert_to_hf_sdxl(te1_state, "text_encoder.")
te2_hf = convert_to_hf_sdxl(te2_state, "text_encoder_2.")
hf_lora.update(te1_hf)
hf_lora.update(te2_hf)

hf_filename = "sdxl_cipher_hf_diffusers.safetensors"
save_file(hf_lora, hf_filename)
print(f"  > Saved: {hf_filename} ({len(hf_lora)} keys)")

print("\n[SUCCESS] Pipeline Complete! Both formats generated.")