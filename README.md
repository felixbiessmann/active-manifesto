# active-manifesto
An Active Learning approach for the manifesto project

## Installation
Create virtualenv:

  `python3.6 -m venv venv`

Activate virtualenv:

  `source venv/bin/activate`

Install dependencies:

  `pip install -r requirements.txt`

Add API-Key to `manifesto_data.py` ([see Manifesto Project Website](https://manifestoproject.wzb.eu/information/documents/api)).

## Docker setup

To build and run the twitter-app image:

```
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
docker build -f Dockerfile_crawler -t crawler .
docker run crawler
```
