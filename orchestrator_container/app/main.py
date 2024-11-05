from fastapi import FastAPI, UploadFile, File
import shutil
import docker
import json
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI()

dir_dataset = Path('/app/shared_data/dataset/')
dir_models = Path('/app/shared_data/models/')


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/uploadtrain/")
async def upload_file(file: UploadFile = File(...)):
    dir_dataset.mkdir(exist_ok=True, parents=True)

    path_dest_file = dir_dataset / 'train.csv'
    if Path(file.filename).suffix != '.csv':
        return {"filename": file.filename, "message": "Only .csv files are accepted"}

    with path_dest_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "message": "File uploaded successfully"}


@app.post("/uploadtest/")
async def upload_file(file: UploadFile = File(...)):
    dir_dataset.mkdir(exist_ok=True, parents=True)

    path_dest_file = dir_dataset / 'test.csv'
    if Path(file.filename).suffix != '.csv':
        return {"filename": file.filename, "message": "Only .csv files are accepted"}

    with path_dest_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "message": "File uploaded successfully"}



@app.get("/train")
async def trainModel():
    client = docker.from_env()

    # TODO: Find a better way of passing the volume. Maybe creating a dedicated volume,
    # not binded to the host.

    name_container = "ml-trainer"
    client.containers.run(
        "ml-trainer:latest",
        name="ml-trainer",
        volumes=["/home/arthur/Projects/MLOpsTraining/shared_data/:/app/shared_data/"]
    )

    container = client.containers.get(name_container)
    container.remove()

    model_metrics = None
    metrics_file = dir_models / 'model_metrics.json'
    with open(metrics_file) as json_file:
        data = json.load(json_file)
        model_metrics = data

    return model_metrics


@app.get("/predict/{htype}_{sector}_{net_usable_area}_{net_area}_{n_rooms}_{n_bathroom}_{latitude}_{longitude}")
async def predict(
    htype: str,
    sector: str,
    net_usable_area: float,
    net_area: float,
    n_rooms: float,
    n_bathroom: float,
    latitude: float,
    longitude: float):

    try:
        net_usable_area = float(net_usable_area)
        net_area = float(net_area)
        n_rooms = float(n_rooms)
        n_bathroom = float(n_bathroom)
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        return {'Error':'ValueError: some numeric value could not be converted to float'}

    supported_house_types = ["departamento", "casa"]
    if not (htype in supported_house_types):
        return {'Error':'ValueError: unkown property type'}

    supported_sectors = ["vitacura", "la reina", "lo barnechea", "providencia", "las condes"]
    if not (sector in supported_sectors):
        return {'Error':'ValueError: unkown sector'}

    # TODO: store columns and supported categorical strings in file during model training
    df = pd.DataFrame(columns=['type', 'sector', 'net_usable_area', 'net_area', 'n_rooms', 'n_bathroom', 'latitude', 'longitude'])
    df.loc[0] = [htype, sector, net_usable_area, net_area, n_rooms, n_bathroom, latitude, longitude]

    path_model = dir_models / 'trained_model.pkl'

    try:
        model = joblib.load(path_model)
    except FileNotFoundError:
        return {'Error' : 'Ensure to have a trained model before performing predictions.'}

    prediction = model.predict(df)[0]

    return {'prediction': prediction}
