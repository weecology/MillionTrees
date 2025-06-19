import os
import glob

tree_locations_dir = "/orange/ewhite/DeepForest/AutoArborist/auto_arborist_cvpr2022_v0.27/tree_locations"
script_path = "/home/b.weinstein/MillionTrees/data_prep/AutoArborist.py"
output_dir = "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery"

csv_files = [f for f in os.listdir(tree_locations_dir) if f.endswith('.csv') and "edmonton" not in f.lower()]

# Remove those labeled train or test
csv_files = [f for f in csv_files if not any(x in f.lower() for x in ["train", "test"])]

for csv_file in csv_files:
    csv_path = os.path.join(tree_locations_dir, csv_file)
    job_name = os.path.splitext(csv_file)[0]
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/{job_name}.out
#SBATCH --error={output_dir}/{job_name}.err
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