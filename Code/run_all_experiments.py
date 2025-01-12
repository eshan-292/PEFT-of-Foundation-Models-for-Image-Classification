# run_all_experiments.py

import subprocess
import os
import sys
import datetime

def run_experiment(script_path, log_dir, results_dir):
    """
    Runs a Python experiment script and logs its output.

    Args:
        script_path (str): Path to the experiment script.
        log_dir (str): Directory to save log files.
        results_dir (str): Directory to save experiment-specific results.

    Returns:
        bool: True if the experiment ran successfully, False otherwise.
    """
    script_name = os.path.basename(script_path)
    experiment_name = os.path.splitext(script_name)[0]
    log_file_path = os.path.join(log_dir, f"{experiment_name}.log")
    experiment_results_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(experiment_results_dir, exist_ok=True)

    with open(log_file_path, 'w') as log_file:
        try:
            print(f"Starting {script_name}...")
            # Run the script as a subprocess
            subprocess.run(
                [sys.executable, script_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True
            )
            print(f"{script_name} completed successfully.\n")
            return True
        except subprocess.CalledProcessError:
            print(f"{script_name} failed. Check the log file at {log_file_path} for details.\n")
            return False

def main():
    # Define the list of experiment scripts in the order to run
    experiment_scripts = [
        "E1.py",
        "E2.py",
        "E3.py",
        "E4.py",
        "E5.py",
        "E6.py"
    ]

    # Create timestamped directories for logs and results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("experiment_logs", f"logs_{timestamp}")
    results_dir = os.path.join("experiment_results", f"results_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Iterate through each experiment script and run it
    for script in experiment_scripts:
        script_path = os.path.join(os.getcwd(), script)
        if not os.path.isfile(script_path):
            print(f"Script {script} not found in the current directory. Skipping.\n")
            continue

        # Run the experiment and log output
        success = run_experiment(script_path, log_dir, results_dir)

        # Optionally, move or copy result files to the experiment-specific directory
        # This assumes that each experiment script saves its outputs in a consistent manner
        # For example, plots are saved with a prefix matching the experiment name

        if success:
            
            print("Experiment completed successfully")
            # # Example: Move all files starting with the experiment name to its results directory
            # for file in os.listdir(os.getcwd()):
            #     if file.startswith(os.path.splitext(script)[0]):
            #         src_path = os.path.join(os.getcwd(), file)
            #         dst_path = os.path.join(results_dir, file)
            #         os.rename(src_path, dst_path)
        else:
            print(f"Experiment {script} did not complete successfully.\n")

    print("All experiments have been executed.")

if __name__ == "__main__":
    main()
