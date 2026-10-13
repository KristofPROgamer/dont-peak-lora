from safetensors.torch import load_file, save_file

peft_file = "DPC_CLIP_LoRA.safetensors"
out_file = "DPC_CLIP_LoRA_Comfyui.safetensors"

# Load the Hugging Face PEFT weights
state_dict = load_file(peft_file)
new_state_dict = {}

for key, weight in state_dict.items():
    # Only process the patched text encoder layers
    if not key.startswith("base_model.model."):
        continue
        
    # Strip PEFT wrapper and replace with standard Diffusers text prefix
    clean_key = key.replace("base_model.model.", "text_model.")
    
    parts = clean_key.split(".")
    lora_suffix = parts[-2]  # "lora_A" or "lora_B"
    layer_path = parts[:-2]  # ["text_model", "encoder", "layers", ...]
    
    # ComfyUI expects layers joined by underscores and prefixed with 'lora_te_'
    comfy_layer_name = "lora_te_" + "_".join(layer_path)
    
    # Map PEFT A/B matrices to ComfyUI down/up matrices
    if lora_suffix == "lora_A":
        comfy_key = f"{comfy_layer_name}.lora_down.weight"
    elif lora_suffix == "lora_B":
        comfy_key = f"{comfy_layer_name}.lora_up.weight"
    else:
        continue
        
    new_state_dict[comfy_key] = weight

save_file(new_state_dict, out_file)
print(f"Successfully converted {len(new_state_dict)} tensors.")
print(f"File saved as: {out_file}")