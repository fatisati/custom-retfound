import subprocess
import re

def parse_qos_output(output):
    """
    Parse the output of the 'sacctmgr show qos' command to extract QoS information.

    Args:
        output (str): The raw command output.

    Returns:
        dict: A dictionary with QoS names as keys and their attributes as values.
    """
    qos_dict = {}
    for line in output.splitlines():
        if not line.strip() or "gpu" not in line:  # Skip empty lines or lines without 'gpu'
            continue
        
        columns = line.split()
        if len(columns) < 2:  # Ensure there are enough columns to process
            print(f"Skipping malformed line: {line}")
            continue
        
        qos_name = columns[0]
        max_jobs_per_user = columns[1]
        max_tres = columns[2] if len(columns) > 2 else ""
        
        # Extract CPU and memory limits from MaxTRES
        cpu = re.search(r'cpu=(\d+)', max_tres)
        mem = re.search(r'mem=([\dA-Za-z]+)', max_tres)
        qos_dict[qos_name] = {
            "max_jobs_per_user": int(max_jobs_per_user) if max_jobs_per_user.isdigit() else None,
            "cpu": int(cpu.group(1)) if cpu else None,
            "memory": mem.group(1) if mem else None,
            "partition": []  # Placeholder for partition data
        }
    return qos_dict

def fetch_partitions_for_qos():
    """
    Fetch the QoS-partition relationship using the 'scontrol show partition' command.

    Returns:
        dict: A dictionary mapping QoS names to their partitions.
    """
    partitions_dict = {}
    command = ["scontrol", "show", "partition"]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if process.returncode != 0:
        print(f"Error running scontrol command: {process.stderr}")
        return partitions_dict
    
    partition_name = None
    for line in process.stdout.splitlines():
        if line.startswith("PartitionName="):
            partition_name = line.split("=")[1].strip()
        
        if "AllowQos=" in line:
            qos_list = line.split("AllowQos=")[1].strip()
            qos_names = qos_list.split(",")
            for qos in qos_names:
                qos = qos.strip()
                
                if partition_name:
                    partitions_dict[qos] = partition_name
    
    return partitions_dict

def filter_valid_qos(qos_dict):
    """
    Filter QoS dictionary to only include entries where all keys have non-None values.

    Args:
        qos_dict (dict): The original QoS dictionary.

    Returns:
        dict: The filtered QoS dictionary.
    """
    return {
        qos_name: attributes
        for qos_name, attributes in qos_dict.items()
        if all(value is not None and value != [] for value in attributes.values())
    }

def generate_qos_file(qos_dict, output_file="qos.py"):
    """
    Generate a Python file with a dictionary of QoS information.

    Args:
        qos_dict (dict): The QoS information to write.
        output_file (str): The file to save the Python dictionary.
    """
    with open(output_file, "w") as f:
        f.write("# Auto-generated QoS dictionary\n\n")
        f.write("qos_data = ")
        f.write(repr(qos_dict))
    print(f"QoS dictionary saved to {output_file}")

def main():
    # Run the sacctmgr command
    command = ["sacctmgr", "show", "qos", "format=Name%-50,MaxJobsPerUser,MaxTRES%-50,MaxWall"]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if process.returncode != 0:
        print(f"Error running sacctmgr command: {process.stderr}")
        return
    
    # Parse QoS data
    output = process.stdout
    qos_dict = parse_qos_output(output)
    
    # Fetch partitions for each QoS
    partitions_dict = fetch_partitions_for_qos()
    
    # Add partitions to QoS dictionary
    for qos_name, partitions in partitions_dict.items():
        if qos_name in qos_dict:
            qos_dict[qos_name]["partition"] = partitions
    
    # Filter QoS dictionary to include only valid entries
    filtered_qos_dict = filter_valid_qos(qos_dict)
    
    # Generate qos.py file
    generate_qos_file(filtered_qos_dict)

if __name__ == "__main__":
    main()
