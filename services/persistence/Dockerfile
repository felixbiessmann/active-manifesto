#FROM ubuntu:16.04
FROM python:3.6-slim

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y sqlite3 libsqlite3-dev
RUN mkdir /db
RUN /usr/bin/sqlite3 ${DB_PATH}

WORKDIR /api
ADD . /api

RUN pip install -r requirements.txt

CMD ["python", "web_api.py"]
