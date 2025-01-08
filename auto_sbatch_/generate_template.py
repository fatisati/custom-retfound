import re

def generate_template_with_param_key(command):
    """
    Generate a template string by replacing parameter values with {param_key}, 
    handling both `--param value` and `--param=value` formats.
    
    Args:
        command (str): The command string with parameters.
        
    Returns:
        str: The command string with parameter values replaced by placeholders in the form {param_key}.
    """
    # Replace `--param=value` format
    command = re.sub(r'(--\w+)=\S+', lambda m: f"{m.group(1)}={{" + m.group(1).lstrip('--') + "}", command)
    # Replace `--param value` format
    command = re.sub(r'(--\w+)\s+\S+', lambda m: f"{m.group(1)} {{" + m.group(1).lstrip('--') + "}", command)
    
    return command

def extract_default_values(command):
    """
    Extract default values for parameters from the command string.
    
    Args:
        command (str): The original command string with parameters.
        
    Returns:
        dict: A dictionary of parameter keys and their default values.
    """
    defaults = {}
    # Match `--param=value` format
    matches = re.findall(r'--(\w+)=(\S+)', command)
    for key, value in matches:
        defaults[key] = value

    # Match `--param value` format
    matches = re.findall(r'--(\w+)\s+(\S+)', command)
    for key, value in matches:
        defaults[key] = value

    return defaults

def save_defaults_and_constants(defaults, output_path="defaults.py", constants_path="constants.py"):
    """
    Save a dictionary as a Python file with a variable named 'defaults' and
    create another file with a dictionary named 'ABBREVIATION' where each key's
    value is the same as the key.

    Args:
        defaults (dict): Dictionary to save.
        output_path (str): Path to save the defaults.py file.
        constants_path (str): Path to save the constants.py file.
    """
    def cast_value(value):
        """
        Try to cast a string value to an integer or float if possible.
        """
        if isinstance(value, str):
            try:
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
        return value

    # Recursively cast values in the dictionary
    def cast_dict(d):
        """
        Recursively cast all values in a dictionary.
        """
        if isinstance(d, dict):
            return {k: cast_dict(cast_value(v)) for k, v in d.items()}
        elif isinstance(d, list):
            return [cast_dict(cast_value(item)) for item in d]
        return d

    # Cast values in the defaults dictionary
    casted_defaults = cast_dict(defaults)

    # Write the defaults dictionary to a Python file
    with open(output_path, 'w') as file:
        file.write("# Auto-generated defaults.py\n\n")
        file.write("defaults = ")
        file.write(repr(casted_defaults))
        file.write("\n")
    print(f"Defaults dictionary saved to: {output_path}")

    # Create the ABBREVIATION dictionary
    abbreviation_dict = {key: key for key in defaults.keys()}

    # Write the ABBREVIATION dictionary to a constants.py file
    with open(constants_path, 'w') as file:
        file.write("# Auto-generated constants.py\n\n")
        file.write("ABBREVIATION = ")
        file.write(repr(abbreviation_dict))
        file.write("\n")
    print(f"ABBREVIATION dictionary saved to: {constants_path}")




def create_sbatch_template_and_defaults(base_template_path, command, sbatch_output_path, defaults_output_path):
    """
    Create a new SBATCH template and generate a defaults.py file.
    
    Args:
        base_template_path (str): Path to the base SBATCH template file.
        command (str): The command string with parameters to append.
        sbatch_output_path (str): Path to save the new SBATCH template file.
        defaults_output_path (str): Path to save the defaults.py file.
    """
    # Load the base template
    with open(base_template_path, 'r') as base_file:
        base_template = base_file.read()
    
    # Generate the command template
    command_template = generate_template_with_param_key(command)
    
    # Append the command template to the base template
    new_template = base_template.strip() + '\n\n# Appended Command\n' + command_template
    
    # Save the new SBATCH template
    with open(sbatch_output_path, 'w') as sbatch_file:
        sbatch_file.write(new_template)
    print(f"New SBATCH template saved to: {sbatch_output_path}")
    
    # Extract defaults and save to defaults.py
    defaults = extract_default_values(command)
    save_defaults_and_constants(defaults, defaults_output_path)

# Example usage
base_template_path = "./templates/base_template.sbatch"
sbatch_output_path = "./templates/generated_template.sbatch"
defaults_output_path = "./templates/defaults.py"
command = """python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ./IDRiD_data/ \
    --task ./finetune_IDRiD/ \
    --finetune ./RETFound_cfp_weights.pth \
    --input_size 224"""

# Create the new SBATCH template and defaults.py
create_sbatch_template_and_defaults(base_template_path, command, sbatch_output_path, defaults_output_path)
