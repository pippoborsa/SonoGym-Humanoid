# SonoGym-Humanoid
## Installation

This project is built on top of the [SonoGym framework](https://github.com/SonoGym/SonoGym). Please first install SonoGym by following the official installation instructions provided in its repository README.  

After completing the SonoGym installation, add the files from this repository into the corresponding pre-existing folders of your local `SonoGym` directory. In particular:

- `tools/...` → merge into `SonoGym/tools/...`
- `source/spinal_surgery/spinal_surgery/tasks/...` → merge into `SonoGym/source/spinal_surgery/spinal_surgery/tasks/...`
- `source/spinal_surgery/spinal_surgery/assets/...` → merge into `SonoGym/source/spinal_surgery/spinal_surgery/assets/...`
- `source/spinal_surgery/spinal_surgery/scenes/...` → merge into `SonoGym/source/spinal_surgery/spinal_surgery/scenes/...`

More generally, each folder in this repository should be merged with the folder having the same path inside `SonoGym`.

## Assets

Download the additional humanoid assets from:

`https://huggingface.co/datasets/Pippoborsa/SonoGym_Humanoid_assets/tree/main`  
Then place them inside:

`SonoGym/source/spinal_surgery/spinal_surgery/assets/data`

by merging the existing folder with the new one.
