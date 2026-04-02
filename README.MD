This is the first commit for the ForceWM. This research is being done by Rahul Krishna, advised by
William Chong and Oussama Khatib

Week 1 goals: Have elementary encoders for each of the modality types(For images run VJEPA, for GEOMETRIES run point cloud)
and for force data run some other encoder

Set up all the different insertion tasks that I want to train in SIM. 

## Setup

1. Create the Conda environment from `environment.yml`:
   `conda env create -f environment.yml`
2. Activate the environment:
   `conda activate opensai-dev`
3. Run the scripts in `installation_scripts/`:
   `bash installation_scripts/install_dependencies.sh`
   `bash installation_scripts/install_core_libraries.sh`

After that, you should be good to go.
