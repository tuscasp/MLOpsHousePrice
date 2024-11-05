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


# Developing

``` bash
docker run -d --name ml-orchestrator -p 80:80 -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd)/shared_data/:/app/shared_data/ ml-orchestrator:latest
docker run --name ml-trainer -v $(pwd)/shared_data/:/app/shared_data/ ml-trainer:latest
```

Attach to a container:
``` bash
docker run -d --name trainer_bain ml-trainer:latest /bin/bash -c "sleep infinity"
docker exec -it trainer_bain /bin/bash
```
