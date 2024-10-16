# Far-Field Image-Based Traversability Mapping for A Priori Unknown Natural Environments

While navigating unknown environments, robots rely primarily on proximate features for guidance in decision making, such as depth information from lidar or stereo to build a costmap, or local semantic information from images. The limited range over which these features can be used may result in poor robot behavior when assumptions about the cost of the map beyond the range of proximate features misguide the robot. Integrating "far-field" image features that originate beyond these proximate features into the mapping pipeline has the promise of enabling more intelligent and aware navigation through unknown terrain. To navigate with far-field features, key challenges must be overcome. As far-field features are typically too distant to localize precisely, they are difficult to place in a map. Additionally, the large distance between the robot and these features makes connecting these features to their navigation implications more challenging. We propose *FITAM*, an approach that learns to use far-field features to predict costs to guide navigation through unknown environments from previous experience in a self-supervised manner. Unlike previous work, our approach does not rely on flat ground plane assumptions or other range sensors to localize observations. We demonstrate the benefits of our approach through simulated trials and real-world deployment on a Clearpath Robotics Warthog navigating through a forest environment.


# Requirements
Requires Docker, nvidia-docker installed.

To run the docker environment, run the following command:
`docker compose run fitam`

## Data
The data used for our simulated experiments can be found [here](https://drive.google.com/drive/folders/1H4AjAfGtS2qcKoP4iL32NAUBd4-wQmC6?usp=sharing). To use this data, download it and place all nested folders in the `results` folder. (E.g., `results/maps`, `results/evaluation_requests`, etc.)

# Swath Generation
Swath libraries are used to quickly convert a position and orientation in a map to the set of indexes covered by an observation (e.g., local observation, radial bin).

Inputs:
- Radial map config
Outputs: 
- Swath library
To build an individual swath library, run the following command:
`python -m fitam.maps.costmap_swath_library -c <path to config> -o <output path>`
To build all swath libraries, run the following command:
`doit -n 10 swath*

# Evaluation Requests

Inputs: 
- Evaluation request config
- Map
Outputs:
- Evaluation request of some number of feasible start-goal locations with a minimum distance between start/goal as defined by the config. 
`doit -n 10 eval_request_*`


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
  - [x] Evaluation requests
  - [ ] Images
  - [ ] Datasets
  - [ ] Trained models
  - [ ] Sampled locations
- [ ] be able to run evaluations
- [ ] Add a description of the project
- [ ] Make maps and other initial information available online 