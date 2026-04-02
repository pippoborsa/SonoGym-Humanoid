# SonoGym-Humanoid
## Installation

This project is built on top of the [SonoGym framework](https://github.com/SonoGym/SonoGym). Please first install SonoGym by following the official installation instructions provided in its repository README.  

After completing the SonoGym installation, add the files from this repository into the corresponding pre-existing folders of your local `SonoGym` directory. In particular:

- `tools/...` → merge into `SonoGym/tools/...`
- all folders under `source/spinal_surgery/spinal_surgery/...` → merge into the corresponding folders under `SonoGym/source/spinal_surgery/spinal_surgery/...`
- 
Merge with:

```bash
rsync -av /SonoGym-Humanoid-main/source/spinal_surgery/spinal_surgery/{Folder}/ /SonoGym/source/spinal_surgery/spinal_surgery/{Folder}
'''

More generally, each folder in this repository should be merged with the folder having the same path inside `SonoGym`.

## Assets

Download the additional humanoid assets from:

`https://huggingface.co/datasets/Pippoborsa/SonoGym_Humanoid_assets/tree/main`  (https://huggingface.co/datasets/Pippoborsa/SonoGym_Humanoid_assets/tree/main)

Then place them inside:

`SonoGym/source/spinal_surgery/spinal_surgery/assets/data`

by merging the existing folder with the new one.

## Pink

Download the pink IK library with:
'conda install -c conda-forge pink'

The project was tested with Pink v3.4.0, although newer versions should also work.

## Training and Testing

After completing the installation and merging this repository into `SonoGym`, training and testing can be run through the standard Isaac Lab / SonoGym workflow.

The two task IDs, to be passed through the `--task` CLI argument, are:

- `Isaac-Robot-US-Guidance-G1-v0`
- `Isaac-robot-US-guided-surgery-G1-pink-v0`

Use the corresponding task ID depending on whether you want to run the ultrasound navigation task or the ultrasound-guided surgery task.

Example:
''python workflows/skrl/train.py --task Isaac-Robot-US-Guidance-G1-v0 --headless''

To run in headless mode, add the headless option to the command line. This disables rendering and is recommended for training, especially when using multiple environments.

## Setup

Task-specific settings can be modified through the corresponding YAML configuration files in 
`SonoGym/source/spinal_surgery/spinal_surgery/tasks/{task name}/cfgs/{task name}.yaml`

In particular, the robot model can be changed directly from the YAML file by switching 'g1' and 'h1' in 'robot/type'.

