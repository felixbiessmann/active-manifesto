#!/usr/bin/env bash

sudo docker rmi services_user_interface -f
sudo docker rmi services_persistence -f
#sudo docker container prune

export WZB_API_KEY=abc
python data_import.py
sudo docker-compose up

