#!/bin/bash

### docker installation
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt-get -y install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    htop \
    python-virtualenv \
    sqlite3

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get -y install docker-ce
sudo groupadd docker
sudo usermod -aG docker $USER
# access with docker ps might not work, relog to machine

### docker compose installation
sudo curl -L https://github.com/docker/compose/releases/download/1.17.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version

### python local setup
virtualenv -p python3 --no-site-packages venv3
. venv3/bin/activate
pip install -U pandas

### clone project and run services
git clone https://github.com/felixbiessmann/active-manifesto
cd active-manifesto/services
./run.sh
