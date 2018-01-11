#!/usr/bin/env bash

# sudo docker stop $(sudo docker ps -a -q)
# sudo docker rm $(sudo docker ps -a -q)
# #
# sudo docker rmi services_user_interface -f
# sudo docker rmi services_manifesto_model -f
# sudo docker rmi news_crawler -f
# sudo docker container prune
# sudo docker-compose build

echo ""

export WZB_API_KEY="9cd9104a725f26bcae04da3eed6bdd40"

python data_import.py

sudo docker-compose up
