import os
import subprocess

def install_grounding_dino():
    git_url = "https://github.com/IDEA-Research/GroundingDINO.git"
    package_name = "GroundingDINO"

    # Prepare the command to install the package in editable mode
    install_cmd = f"pip install -e git+{git_url}@main#egg={package_name}"

    # Execute the command using subprocess
    try:
        subprocess.check_call(install_cmd, shell=True)
        print(f"{package_name} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}.")

if __name__ == "__main__":
    install_grounding_dino()
