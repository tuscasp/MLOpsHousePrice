# MLOpsTraining

``` bash
docker build -t ml-trainer:latest trainer_container
docker build -t ml-orchestrator:latest orchestrator_container
```

# Release

Copy the dataset to a folder shared with the containers:
```
-repo_root
 ├- shared_data
 |  ├- datasaet
 |  |  ├-test.csv
 |  |  ├- train.csv
 |  ├- models
```

``` bash
docker run -d --name ml-orchestrator -p 80:80 -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd)/shared_data/ /app/shared_data/ ml-orchestrator:latest
docker run --name ml-trainer -v $(pwd)/shared_data/ /app/shared_data/ ml-orchestrator:latest
```

# Developing

Attach to a container:
``` bash
docker run -d --name trainer_bain ml-trainer:latest /bin/bash -c "sleep infinity"
docker exec -it trainer_bain /bin/bash
```
