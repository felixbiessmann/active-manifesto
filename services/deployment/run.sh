#!/usr/bin/env bash

#sudo docker rmi services_user_interface -f
#sudo docker rmi services_persistence -f
#sudo docker rmi services_manifesto_model -f
#sudo docker container prune

echo ""

export WZB_API_KEY="9cd9104a725f26bcae04da3eed6bdd40"

python data_import.py

sudo docker-compose up
