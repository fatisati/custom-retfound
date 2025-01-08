import os
import subprocess
import itertools

from interpretable_ssl.configs.defaults import *
import time
from interpretable_ssl.model_name import generate_model_name

from constants import *
from name import generate_model_name
from defeaults import defaults_dict
from qos import qos_data


class ExperimentRunner:
    def __init__(
        self,
        user='e-helmholtz',
        template_file="./templates/generated_template.sbatch",
        output_dir="./out_files/",
        sbatch_dir="./sbatch_files/",
    ):
        self.user = user
        self.template_file = template_file
        self.output_dir = output_dir
        self.sbatch_dir = sbatch_dir

        self.defaults = defaults_dict.copy()
        self.original_defaults = self.defaults.copy()
        self.qos_dict = qos_data

    def count_jobs(self, qos_name):
        try:
            # Construct the command
            command = f"squeue -u {self.user} -O qos,state | grep {qos_name} | wc -l"

            # Execute the command and capture the output
            result = subprocess.check_output(command, shell=True, text=True)

            # Convert result to an integer
            running_jobs = int(result.strip())

            return running_jobs

        except subprocess.CalledProcessError as e:
            # Handle errors (e.g., if the command fails)
            print(f"Error executing command: {e}")
            return None

    def get_best_qos(self):
        """
        Returns the QoS with available capacity (i.e., where the number of running jobs is less than the max allowed).
        If no such QoS exists, return None.
        """
        for qos_name in self.qos_dict.keys():
            max_jobs = self.qos_dict[qos_name]["max_jobs_per_user"]
            running_jobs = self.count_jobs(qos_name)

            if running_jobs is not None and running_jobs < max_jobs:
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
        qos_resource = get_qos_resources(best_qos)

        # Update the defaults dictionary with the best QoS
        self.defaults["qos"] = best_qos
        self.defaults["memory"] = self.qos_dict[best_qos]["memory"]
        self.defaults["cpus_per_task"] = self.qos_dict[best_qos]["cpu"]
        self.defaults["workers"] = self.qos_dict[best_qos]["cpu"] - 1
        
    def generate_sbatch_content(self):
        """Generate the content for the SBATCH file based on the current parameters."""

        # self.update_qos
        # Read the SBATCH template file
        file_path = os.path.join(self.sbatch_dir, self.template_file)
        with open(file_path, "r") as file:
            sbatch_content = file.read()
        # Return the formatted content with the updated defaults
        return sbatch_content.format(**self.defaults)

    def run_experiment(self):
        """Generate the SBATCH file, save it, and submit the job."""
        job_name = self.defaults["job_name"]
        experiment_dir = os.path.join(self.output_dir, job_name)

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
                ["squeue", "--user", self.user, "--format", "%.200j"],
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

    def run_multiple_experiments(self, item_to_test, submit=False):
        """Run or save multiple experiments by varying parameters according to the item_to_test."""
        keys, values = zip(*item_to_test.items())

        for value_combination in itertools.product(*values):
            params = dict(zip(keys, value_combination))
            
            data_path, finetune = get_data_model(params['modality'], params['experiment'])
            # Generate job name automatically based on differences from defaults
            params['data_path'] = data_path
            params['finetune'] = finetune
            
            job_name = generate_model_name(self.original_defaults.copy(), params)
            params['task'] = os.path.join(self.output_dir, job_name)
            

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
        sbatch_file = os.path.join(
            self.sbatch_dir, "auto_generated", f"run_{job_name}.sbatch"
        )
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


# modality -> op, opt: data_path, finetune
# balance, not: balance
# lossL loss

if __name__ == "__main__":
    print("runner started...")
    runner = ExperimentRunner("swav_template.sbatch")
    experiments = [
        {
            'experiment': ['retfound'],
            'modality': ['opt', 'op'],
            'stats_source': ['custom'],
            'more_augmentation': [0, 1]
        }
    ]
    for item_to_test in experiments:
        runner.run_multiple_experiments(item_to_test, False)
