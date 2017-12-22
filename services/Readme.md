# orchestration of services

For the orchestration we use `docker-compose`, to install it on any linux machine:

```
sudo curl -L https://github.com/docker/compose/releases/download/1.17.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

To startup all services, simply `$ ./run.sh`. This command will setup the initial
database and run all containers, defined in `docker-compose.yml`.

When the containers have started, visit `http://localhost:8080` for the UI.
Once the persistence container has started it's web api you can request samples
and submit user labels.


# EC2 deployment

```
export kn=tdhd
iid=$(aws ec2 run-instances --image-id ami-df8406b0 --security-group-ids launch-wizard-1 --count 1 --instance-type t2.micro --key-name $kn --query 'Instances[0].InstanceId' | sed -e 's/^"//' -e 's/"$//')
while true
do
   echo "Checking EC2 for public IP..."
   ip=$(aws ec2 describe-instances --instance-ids $iid --query 'Reservations[0].Instances[0].PublicIpAddress' | sed -e 's/^"//' -e 's/"$//')
   aws ec2 describe-instances --instance-ids $iid --query 'Reservations[0].Instances[0].State.Name' # "running"
   if (($#ip > 5)); then
     echo "EC2 instance is up @ $ip"
     break;
   fi
   sleep 2
done
ssh -i ~/.ssh/${kn}.pem ubuntu@$ip
```

## build and test single containers

### manifesto model

trigger training with

```
curl -XPOST -H 'Content-Type: application/json' "localhost:5000/train" -d '{"data": [{"label": "left", "text": "meh meh"}, {"label": "right", "text": "3"}, {"label": "right", "text": "3"}, {"label": "right", "text": "3"}, {"label": "right", "text": "3"}, {"label": "left", "text": "3"}, {"label": "left", "text": "3"}, {"label": "left", "text": "3"}, {"label": "left", "text": "3"}]}'
```

get ordering of texts

```
curl -XPOST -H 'Content-Type: application/json' "localhost:5000/estimate_uncertainty" -d '{"data": [{"text_id": 1, "text": "meh meh"}, {"text_id": 2, "text": "3"}]}'
```


### manifesto data container

add wzb manifesto api key in `persistence/Dockerfile` to environment variable.

```
cd persistence
docker build -t persistence .
docker run persistence
```

this will start the manifesto data download and sqlite3 import.
the data should be accessd via HTTP, e.g:

```
curl "localhost:8888/texts?n=300"
# group by label, count
curl "localhost:8888/texts?n=300" | jq '[{label: .data[].label}]' | jq 'group_by(.label)' | jq 'map({n_texts: length, label: .[0].label})' | jq 'sort_by(.n_texts)'
```

will return 300 texts with labels and label sources.


### twitter app

To build and run the twitter-app image:

```
cd apps
docker build -t twitter .
docker run -p 0.0.0.0:80:5000 twitter
```

This will install a `python-3.6` distribution
with all of the requirements and start the web-app
inside the container on port 5000 and forward it to
port 80 on the host machine.

Visit `http://localhost` after you have built and ran the image.

### Installing and running news crawler image

```
cd news_crawler
docker build -f Dockerfile -t crawler .
docker run -p 0.0.0.0:27017:27017 crawler
```
