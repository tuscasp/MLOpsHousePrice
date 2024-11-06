# MLOpsTraining

``` bash
cd <path_to>/MLOpsTraining/
mkdir shared_folder

docker build -t ml-orchestrator:latest orchestrator_container
docker build -t ml-trainer:latest trainer_container

docker run -d --name ml-orchestrator -p 80:80 -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd)/shared_data/:/app/shared_data/ ml-orchestrator:latest

python scripts/upload_dataset.py --train prompt/train.csv --test prompt/test.csv
```

Now that the system is up and running, open a web browser and check out the system outputs for training on uploaded data and for predicting on dummy data:
* http://localhost/train
* http://localhost/predict/casa_vitacura_50.0_40.0_3.0_2.0_-30.0_-40.0

You may get a complete documentation by openning `http://localhost/docs`.


# Assumptions

* Dataset fits in memory
* Train dataset is fixed, i.e. once the model is trained it no longer updates
* There is no need to store multiple models at once
* Dataset is curated, i.e. no missing values

# Next steps

* Run prediction in a dedicated container
* Enable models to be updated with continuous stream of data
* Add support to multiple models and track their relative performance on test dataset
* Improve training pipeline robustness to missing data
