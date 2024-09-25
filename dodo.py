# get the main config!
from fitam.pipeline.sim_experiment_config import pipeline_config
from fitam.pipeline.construct_tasks_from_pipeline import make_tasks_from_config

functions = make_tasks_from_config(pipeline_config, headless=False)
for name, func in functions:
    globals()["task_" + name] = func
