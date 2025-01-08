print('hello before imports')
import os
import subprocess
import itertools

import time
import sys

from auto_sbatch.constants import *
from auto_sbatch.name import generate_model_name
from auto_sbatch.defaults import defaults_dict
from auto_sbatch.qos import qos_data
import socket


class ExperimentRunner:
    def __init__(
        self,
        max_jobs=3,
        user="e-helmholtz",
        template_file="auto_sbatch/templates/generated_template.sbatch",
        output_dir="./out_files/",
        sbatch_dir="./sbatch_files/",
    ):
        self.user = user
        self.template_file = template_file
        self.output_dir = output_dir
        self.sbatch_dir = sbatch_dir

        self.defaults = defaults_dict.copy()
        self.original_defaults = self.defaults.copy()
        self.qos_dict = {
            "jobs-gpu": {"count": max_jobs, "memory": "150G"},
            # "jobs-gpu-long": {"count": 3, "memory": "32"},
        }

        self.used_ports = []
        self.init_port = 48798

    def count_jobs_in_partition(self, partition_name):
        """
        Count the number of jobs for a specific user in a given SLURM partition.

        Args:
            user (str): Username to filter jobs.
            partition_name (str): Name of the SLURM partition.

        Returns:
            int: Number of jobs for the user in the specified partition.
        """
        try:
            # Run the squeue command to get job details
            result = subprocess.run(
                [
                    "squeue",
                    "--user",
                    self.user,
                    "--partition",
                    partition_name,
                    "--noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            # Count lines in the output (each line represents a job)
            job_count = (
                len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
            )
            return job_count

        except subprocess.CalledProcessError as e:
            print(f"Error running squeue: {e.stderr.strip()}")
            return 0

    def jupyter_job_count(self):
        jobs = self.get_current_jobs()
        jupyter_cnts = 0
        for job in jobs:
            if 'jupyter' in job:
                jupyter_cnts += 1
                
        
        return jupyter_cnts
    
    def get_best_qos(self):
        """
        Returns the QoS with available capacity (i.e., where the number of running jobs is less than the max allowed).
        If no such QoS exists, return None.
        """
        for qos_name in self.qos_dict.keys():
            max_jobs = self.qos_dict[qos_name]["count"]
            
            running_jobs = self.count_jobs_in_partition(qos_name)
            jupyter_jobs = self.jupyter_job_count()
            print(f"qos {qos_name}, {running_jobs}, {max_jobs}")
            if running_jobs is not None and running_jobs - jupyter_jobs < max_jobs:
                # Return the first QoS that has available capacity
                return qos_name

        # If no QoS has available capacity, return None
        return None

    def update_params(self, **kwargs):
        self.defaults = self.original_defaults.copy()
        """Update experiment parameters with provided keyword arguments."""
        self.defaults.update(kwargs)

        # Set the output and error file paths based on the job_name
        job_name = self.defaults["job_name"]
        self.defaults["output_file"] = os.path.join(
            self.output_dir, job_name, "output.txt"
        )
        self.defaults["error_file"] = os.path.join(
            self.output_dir, job_name, "error.txt"
        )

    def update_qos(self):
        # Continuously try to get a valid QoS
        best_qos = None
        while not best_qos:
            best_qos = self.get_best_qos()
            if not best_qos:
                print("No available QoS. Retrying...")
                time.sleep(60)  # Wait 10 seconds before retrying

        print(best_qos)

        # Update the defaults dictionary with the best QoS
        self.defaults["qos"] = best_qos
        # self.defaults["memory"] = self.qos_dict[best_qos]["memory"]
        # self.defaults["cpus_per_task"] = self.qos_dict[best_qos]["cpu"]
        # self.defaults["workers"] = self.qos_dict[best_qos]["cpu"] - 1

    def generate_sbatch_content(self):
        """Generate the content for the SBATCH file based on the current parameters."""

        self.update_qos()
        # Read the SBATCH template file
        # file_path = os.path.join(self.sbatch_dir, self.template_file)
        file_path = self.template_file
        with open(file_path, "r") as file:
            sbatch_content = file.read()
        # Return the formatted content with the updated defaults
        return sbatch_content.format(**self.defaults)

    def run_experiment(self):
        """Generate the SBATCH file, save it, and submit the job."""
        job_name = self.defaults["job_name"]
        experiment_dir = os.path.join(self.output_dir, job_name)

        test_file = os.path.join(experiment_dir, 'confusion_matrix_test.jpg')
        if os.path.exists(test_file):
            print(job_name, ' already trained')
            return
        
        # Create directory for job output
        os.makedirs(experiment_dir, exist_ok=True)

        # Generate SBATCH content
        sbatch_content = self.generate_sbatch_content()

        # Write the SBATCH file
        sbatch_file = os.path.join(experiment_dir, f"run_{job_name}.sbatch")
        with open(sbatch_file, "w") as file:
            file.write(sbatch_content)

        # Submit the SBATCH job
        subprocess.run(["sbatch", sbatch_file])

        print(f"Submitted job: {job_name}")

    def get_current_jobs(self):
        try:

            # Run the 'squeue' command for the specified user and capture the output
            result = subprocess.run(
                ["squeue", "--user", self.user, "--format", "%.250j"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Check for errors
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return []

            # Split the output into lines, strip leading and trailing whitespaces
            lines = [line.strip() for line in result.stdout.splitlines()]

            # Skip the header (first line) and return the remaining lines (job names)
            return lines[1:]  # Skip the first line if it contains the header

        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def get_port(self):
        # if len(self.used_ports)==0:
        #     port = self.init_port
        # else:

        #     min_port = min(self.used_ports)
        #     port = min_port - 1
        # self.used_ports.append(port)
        # return port
        return find_free_port()

    def run_multiple_experiments(self, item_to_test, submit=False):
        """Run or save multiple experiments by varying parameters according to the item_to_test."""
        keys, values = zip(*item_to_test.items())

        for value_combination in itertools.product(*values):
            params = dict(zip(keys, value_combination))

            data_path, finetune, nb_classes = get_data_model(
                params["modality"], params["experiment"]
            )
            # Generate job name automatically based on differences from defaults
            params["data_path"] = data_path
            params["finetune"] = finetune
            params["nb_classes"] = nb_classes

            job_name = generate_model_name(
                self.original_defaults.copy(), params, ABBREVIATIONS
            )
            params["task"] = os.path.join(self.output_dir, job_name) + "/"

            params["master_port"] = self.get_port()
            print(params["master_port"])
            current_jobs = self.get_current_jobs()

            if job_name in current_jobs:
                print(job_name, " already running")
                continue
            params["job_name"] = job_name
            # params["experiment_name"] = job_name

            self.update_params(**params)

            if submit:
                self.run_experiment()  # Submit the job
            else:
                self.save_sbatch_file()  # Only save the SBATCH file for review

    def save_sbatch_file(self):
        """Generate and save the SBATCH file without submitting it."""
        job_name = self.defaults["job_name"]
        experiment_dir = os.path.join(self.output_dir, "auto_generated", job_name)

        # Create directory for job output
        os.makedirs(experiment_dir, exist_ok=True)

        # Generate SBATCH content
        sbatch_content = self.generate_sbatch_content()

        # Write the SBATCH file
        sbatch_file = os.path.join(self.sbatch_dir, f"run_{job_name}.sbatch")
        with open(sbatch_file, "w") as file:
            file.write(sbatch_content)

        print(f"Saved SBATCH file: {sbatch_file}")


def evaluate_job_count(items_to_test):
    total_jobs = 0

    for idx, item_to_test in enumerate(items_to_test):
        keys, values = zip(*item_to_test.items())

        # Calculate the number of combinations for this particular item_to_test
        num_combinations = len(list(itertools.product(*values)))

        print(f"Item {idx + 1} will submit {num_combinations} jobs.")
        total_jobs += num_combinations

    print(f"Total number of jobs to be submitted: {total_jobs}")
    return total_jobs


def get_master_port():
    pass


def find_free_port():
    """Find a free port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to an available port provided by the OS
        port = s.getsockname()[1]  # Retrieve the port number
    return port


# modality -> op, opt: data_path, finetune
# balance, not: balance
# lossL loss
print('experiment runner called')

if __name__ == "__main__":
    print('hello from main')
    print("runner started...")
    runner = ExperimentRunner(1)

    experiments = [
        {
            "experiment": ["scivias"],
            "modality": ['opt', 'op'],
            'stats_source': ['custom'],
            'transform': ['custom'],
        },
        {
            "experiment": ["scivias"],
            "modality": ['opt'],
            'stats_source': ['custom'],
            'more_augmentation': [0,1],
            'epochs': [700]
        },
        {
            "experiment": ["scivias"],
            "modality": ['opt', 'op'],
            'stats_source': ['custom', 'imagenet'],
            'more_augmentation': [0,1]
        },
        {
            "experiment": ["retfound"],
            'batch_size': [16],
            'balanced': [0],
            'modality': ['op', 'opt'],
            'stats_source': ['custom', 'imagenet'],
            
        }
    ]
    for item_to_test in experiments:
        runner.run_multiple_experiments(item_to_test, True)
