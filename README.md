# fitam

# Docker Environment
Requires Docker, nvidia-docker installed.

To run the docker environment, run the following command:
`docker compose run fitam`

# Swath Generation
Swath libraries are used to quickly convert a position and orientation in a map to the set of indexes covered by an observation (e.g., local observation, radial bin).

Inputs:
- Radial map config
Outputs: 
- Swath library
To build an individual swath library, run the following command:
`python -m fitam.maps.costmap_swath_library -c <path to config> -o <output path>`
To build all swath libraries, run the following command:
`./scripts/build_swath_libraries.sh`

# Evaluation Requests

# Images

# Datasets

# Trained Models

# Run Evaluations 

Core classes:
mapping/ 
- LandCoverComplexMap
- LandCoverMap
- OccpuancyGrid
- SwathLibrary
- Belief and subclasses
learning/
- Model types
- Dataloaders
- Trainers
generation/
- Image rendering code
core/
- Configs
evaluation/
- planner_trial_opengl
analysis/
- Analysis code


## TODO
- [x] Setup docker environment
be able to generate:  
  - [x] Swaths
  - [ ] Evaluation requests
  - [ ] Images
  - [ ] Datasets
  - [ ] Trained models
  - [ ] Sampled locations
- [ ] be able to run evaluations
- [ ] Add a description of the project
- [ ] Make maps and other initial information avalible online 