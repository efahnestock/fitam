# Far-Field Image-Based Traversability Mapping for *A Priori* Unknown Natural Environments

![sim](images/sim_pano_cropped.gif)

This repository contains code for the simulated results from the paper "Far-Field Image-Based Traversability Mapping for *A Priori* Unknown Natural Environments". The code is organized into a series of tasks that can be run using the `doit` task runner. Please feel free to open an issue/PR if you have any questions or suggestions!

## Abstract
While navigating unknown environments, robots rely primarily on proximate features for guidance in decision making, such as depth information from lidar or stereo to build a costmap, or local semantic information from images. The limited range over which these features can be used may result in poor robot behavior when assumptions about the cost of the map beyond the range of proximate features misguide the robot. Integrating "far-field" image features that originate beyond these proximate features into the mapping pipeline has the promise of enabling more intelligent and aware navigation through unknown terrain. To navigate with far-field features, key challenges must be overcome. As far-field features are typically too distant to localize precisely, they are difficult to place in a map. Additionally, the large distance between the robot and these features makes connecting these features to their navigation implications more challenging. We propose *FITAM*, an approach that learns to use far-field features to predict costs to guide navigation through unknown environments from previous experience in a self-supervised manner. Unlike previous work, our approach does not rely on flat ground plane assumptions or other range sensors to localize observations. We demonstrate the benefits of our approach through simulated trials and real-world deployment on a Clearpath Robotics Warthog navigating through a forest environment.

# Requirements

Requires Docker, nvidia-docker installed.

To run the docker environment, run the following command:
```docker compose run fitam```

To run a jupyter notebook server in the `fitam` environment run:
```TOKEN=your_jupyter_token docker compose run --service-ports notebook```

Confirm your GPU is visible by running `nvidia-smi`. Execute all tasks below in the docker environment. All available tasks can be seen by running `doit list`. 

# Data

The data used for our simulated experiments can be found [here](https://drive.google.com/drive/folders/1H4AjAfGtS2qcKoP4iL32NAUBd4-wQmC6?usp=sharing). To use this data, download it and place all nested folders in the `results` folder. (E.g., `results/maps`, `results/evaluation_requests`, etc.)

## Swath Generation

Swath libraries are used to quickly convert a position and orientation in a map to the set of indexes covered by an observation (e.g., local observation, radial bin).

Inputs:

- Radial map config  

Outputs:
- Swath library

To build an individual swath library, run the following command:

```python -m fitam.maps.costmap_swath_library -c <path to config> -o <output path>```

To build all swath libraries, run the following command:

`doit -n <num_processes> swath*`

## Evaluation Requests

An evaluation request is a set of start-goal locations that are feasible for a planner to navigate. The minimum distance between start/goal locations is defined by the config. Specific to each map. Note evaluation requests are supplied, so the evaluation request generation is commented out in `sim_experiment_config.py` and will need to be uncommented if you want to create new evaluation requests.

Inputs:

- Evaluation request config
- Map

Outputs:

- Evaluation request of some number of feasible start-goal locations with a minimum distance between start/goal as defined by the config.

To run:

```doit -n <num_processes> eval_request_*```

## Timed Sampling

Given a map, navigate through the map to a random set of waypoints. Stops when a maximum cost (in seconds) is accumulated. Produce a partially observed map based on this trajectory. Note timed sampling is supplied, so the evaluation request generation is commented out in `sim_experiment_config.py` and will need to be uncommented if you want to create new evaluation requests.
Inputs:

- Map
- Config (e.g., number of waypoints, distance between waypoints, total navigation time)

Outputs:

- Partially observed map in `MAPS_DIR`
- Image generation request in `SAMPLED_LOCATIONS_DIR`

To run:

```doit timed_sampling*```

## Splitting up Image Requests

Given an image request, subdivide it based on navigation time. E.g., if the image request is from a 60 minute trajectory, subdivide it into 10m, 30m trajectories. Note image requests are supplied, so the evaluation request generation is commented out in `sim_experiment_config.py` and will need to be uncommented if you want to create new evaluation requests.

Inputs:

- Fully observed map
- Image request from trajectory
- Navigation times to split the image request into

Outputs:

- Subdivided image requests

To run:

```python -m fitam.generation.split_image_request_by_time -i results/sampled_locations/training_map/image_request.pkl -m results/maps/final_experiment_train_balt -v 25 -t 10 30 60 120 360 -o split_up_image_requests```

## Rendering Images
Given a map and a set of sampled locations, render panoramas in those locations.

Inputs:

- Fully observed map
- Image request

Outputs:

- Rendered images

To run: 

```doit -n <n_processes> image_rendering*```

## Datasets
Datasets take in a set of rendered images and produce a dataset for training, cropping the images into slices and balancing the dataset.

Inputs:

- Dataset config 
- Rendered images
- Radial map config

Outputs:

- Dataset

To run:

```doit dataset*```

## Train Models
Given a dataset, model config/type, train an ensemble of far-field models.

Inputs:

- Dataset
- Train Config
- Radial Map Config

Outputs:

- Trained models

To run:

```doit training*```


## Run Evaluations
Given a trained model, a map, and an evaluation request, run evaluations on the model.

Inputs:

- Trained model
- Map
- Evaluation request

Outputs:

- Evaluation results

To run:

```doit evaluation*```
