# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Define mountable directories.
VOLUME ["/data/db"]

# Define working directory.
WORKDIR /data

# Copy the current directory contents into the container at /app
ADD . /data

# Install MongoDB. debian sysvinit
RUN \
  apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10 && \
  echo 'deb http://downloads-distro.mongodb.org/repo/debian-sysvinit dist 10gen' > /etc/apt/sources.list.d/mongodb.list && \
  apt-get update && \
  apt-get install -y mongodb-org && \
  rm -rf /var/lib/apt/lists/*

# run mongo daemon
RUN mongod &

# Expose ports.
#   - 27017: process
#   - 28017: http
EXPOSE 27017
EXPOSE 28017

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

CMD ["python", "newsreader.py"]
