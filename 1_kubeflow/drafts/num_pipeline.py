import kfp
from kfp import dsl, compiler

# Define components for Addition and Subtraction
@dsl.component(base_image='python:3.11',packages_to_install=['appengine-python-standard'])
def num_add(a: int, b: int) -> int:
    return a + b

@dsl.component(base_image='python:3.11',packages_to_install=['appengine-python-standard'])
def num_sub(a: int, b: int) -> int:
    return a - b

# Define the second pipeline (num_pipeline)
@dsl.pipeline(name='num_pipeline')
def num_pipeline(a1: int, b1: int) -> int:
    # Define tasks that will be executed in the pipeline
    pipeline_task1 = num_add(a=a1, b=b1)  # Adds a and b
    pipeline_task2 = num_sub(a=a1, b=b1)  # Subtracts a from b
    
    # Complete task will use the outputs of previous tasks, chaining them
    complete_task = num_add(a=pipeline_task1.output, b=pipeline_task2.output)

    # Return the output of the final task (result of addition)
    return complete_task.output

# Initialize the Kubeflow Pipelines client (Replace `kfp_endpoint` with your actual server URL)
kfp_endpoint = None  # For local, or specify your server URL
client = kfp.Client(host=kfp_endpoint)

experiment_name = 'My Number Pipeline Experiment'

# Create and execute the pipeline with given arguments
client.create_run_from_pipeline_func(
    num_pipeline,
    experiment_name=experiment_name,
    arguments={
        "a1": 100,
        "b1": 200
    }
)
