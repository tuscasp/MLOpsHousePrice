from fastapi import FastAPI
import docker
import json

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/train")
async def trainModel():
    client = docker.from_env()
    # TODO: Find a better way of passing the volume. Maybe creating a dedicated volume,
    # not binded to the host.
    client.containers.run(
        "ml-trainer:latest",
        name="ml-trainer",
        volumes=["/home/arthur/Projects/MLOpsTraining/shared_data/:/app/shared_data/"]
    )

    model_metrics = None
    with open('/app/shared_data/models/model_metrics.json') as json_file:
        data = json.load(json_file)
        model_metrics = data

    return model_metrics
