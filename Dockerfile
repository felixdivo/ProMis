FROM ubuntu:22.04

# APT installs and settings
RUN apt-get update -qq
RUN apt-get install -qy git curl xz-utils python3-pip python3-gdal libgdal-dev cython3

# Locales settings for Sphinx to work
RUN apt-get install -qy locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen

# Git and pip setup
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install separate pip dependencies
RUN pip install pyro-ppl graphviz
RUN pip install --upgrade --force-reinstall --no-deps --no-binary :all: pysdd

# Install Problog with distributional clauses
# This specific repo contains bugfixes that are not part of the official release yet
RUN pip install git+https://github.com/simon-kohaut/problog.git@dcproblog_develop

# Get clone of repository
RUN git clone https://github.com/HRI-EU/ProMis.git
WORKDIR /ProMis
# Setting -e does not really work here
RUN pip install '.[doc,dev]'
