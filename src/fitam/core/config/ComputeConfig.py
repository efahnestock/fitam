

from dataclasses import dataclass


@dataclass
class ComputationConfig:
    num_processes: int = 20
    num_gpu_training_instances: int = 5  # number of NN training instances we run in parallel
    num_locations_per_process: int = 200  # number of locations to generate per process for dataset generation
    num_planning_requests_per_process: int = 10  # hwo to tile the planning requests