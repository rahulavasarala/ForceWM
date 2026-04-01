This is the first commit for the ForceWM. This research is being done by Rahul Krishna, advised by
William Chong and Oussama Khatib

Week 1 goals: Have elementary encoders for each of the modality types(For images run VJEPA, for GEOMETRIES run point cloud)
and for force data run some other encoder

Set up all the different insertion tasks that I want to train in SIM. 

## OpenSai Setup

1. Run `bash sai_installation_scripts/install_dependencies.sh` to create or update the Conda environment defined in `environment.yml`.
2. Run `bash sai_installation_scripts/install_core_libraries.sh` to clone and build the OpenSai core libraries against the active Conda environment.

The default environment name is `opensai-dev`. You can override it with `OPENSAI_CONDA_ENV=<name>`.
