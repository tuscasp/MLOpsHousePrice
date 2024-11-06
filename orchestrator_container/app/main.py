from fastapi import FastAPI, UploadFile, File
import shutil
import docker
import json
import joblib
import os
import pandas as pd
from pathlib import Path

import logging
import sys


app = FastAPI()

dir_shared = Path('/app/shared_data')
dir_dataset = dir_shared / 'dataset'
dir_models = dir_shared / 'models'

logs_path = dir_shared / 'server_logs.log'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler(logs_path)  # Log to a file
    ]
)

# Create a custom logger for your application
logger = logging.getLogger("fastapi_app")


@app.get("/")
async def root():
    return {"message": "Welcome to Property-Friends Real State"}

@app.post("/uploadtrain/")
async def upload_file(file: UploadFile = File(...)):
    dir_dataset.mkdir(exist_ok=True, parents=True)

    path_dest_file = dir_dataset / 'train.csv'
    if Path(file.filename).suffix != '.csv':
        return {"filename": file.filename, "message": "Only .csv files are accepted"}

    with path_dest_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info('Train dataset file uploaded')

    return {"filename": file.filename, "message": "File uploaded successfully"}


@app.post("/uploadtest/")
async def upload_file(file: UploadFile = File(...)):
    dir_dataset.mkdir(exist_ok=True, parents=True)

    path_dest_file = dir_dataset / 'test.csv'
    if Path(file.filename).suffix != '.csv':
        return {"filename": file.filename, "message": "Only .csv files are accepted"}

    with path_dest_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info('Test dataset file uploaded')

    return {"filename": file.filename, "message": "File uploaded successfully"}



@app.get("/train")
async def trainModel():
    client = docker.from_env()

    path_shared_dir_host = os.getenv('HOST_SHARED_DIR')

    name_container = "ml-trainer"
    client.containers.run(
        "ml-trainer:latest",
        name="ml-trainer",
        volumes=[f"{path_shared_dir_host}/:/app/shared_data/"]
    )

    container = client.containers.get(name_container)
    container.remove()

    model_metrics = None
    metrics_file = dir_models / 'model_metrics.json'
    with open(metrics_file) as json_file:
        data = json.load(json_file)
        model_metrics = data

    logger.info(f'Model trained successfully with output{model_metrics}')


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
