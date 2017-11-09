#!/usr/bin/env bash

sudo docker rmi services_user_interface -f
sudo docker rmi services_persistence -f
sudo docker container prune

echo ""

export WZB_API_KEY=""
#export HOST_DB_PATH="./db/active-manifesto.db"
#export CONTAINER_DB_PATH="/db/active-manifesto.db"

python data_import.py

sudo docker-compose up
