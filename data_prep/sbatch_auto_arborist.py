import os
import glob

tree_locations_dir = "/orange/ewhite/DeepForest/AutoArborist/auto_arborist_cvpr2022_v0.27/tree_locations"
script_path = "/home/b.weinstein/MillionTrees/data_prep/AutoArborist.py"
output_dir = "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery"

# Remove those labeled train or test
csv_files = [os.path.join(tree_locations_dir, f) for f in os.listdir(tree_locations_dir) if f.endswith('.csv')]
csv_files = [f for f in csv_files if not any(x in f.lower() for x in ["train", "test"])]

# basename of csv_files
csv_files = [os.path.basename(f) for f in csv_files]

# Just do one for now
csv_files = ["ColumbusTrees.csv"]

for csv_file in csv_files:
    # If output annotations already exist, skip
    if os.path.exists(os.path.join(output_dir, f"{os.path.basename(csv_file).lower().replace('.csv', '')}_annotations.csv")):
        continue
    
    csv_path = os.path.join(tree_locations_dir, csv_file)
    job_name = os.path.splitext(csv_file)[0]
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=/home/b.weinstein/logs/{job_name}.out
#SBATCH --error=/home/b.weinstein/logs/{job_name}.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite

module load anaconda
source activate MillionTrees

python {script_path} "{csv_path}" --output_dir "{output_dir}"
"""
    sbatch_path = os.path.join(output_dir, f"sbatch_{job_name}.sh")
    with open(sbatch_path, "w") as f:
        f.write(sbatch_script)
    os.system(f"sbatch {sbatch_path}")