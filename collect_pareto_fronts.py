import os
import glob
import shutil

root_dir = 'simulation'
target_dir = 'all_pareto_fronts'
os.makedirs(target_dir, exist_ok=True)

# Limit to run_1 through run_50
for i in range(1, 51):
    run_folder = f'run_{i}'
    pareto_path = os.path.join(root_dir, run_folder, 'pareto_front')
    if os.path.isdir(pareto_path):
        txt_files = glob.glob(os.path.join(pareto_path, '*.txt'))
        for file_path in txt_files:
            base_name = os.path.basename(file_path)
            new_name = f"{run_folder}_{base_name}"
            new_path = os.path.join(target_dir, new_name)
            try:
                shutil.copy(file_path, new_path)
            except Exception as e:
                print(f"Error copying {file_path}: {e}")

print("Selected Pareto front files have been collected.")
