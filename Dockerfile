FROM python:3.8-slim-buster

# NECCESARY FOR igraph
RUN apt-get -qq update
# RUN apt-get install -y libigraph0-dev
RUN apt-get install -y cmake

# INSTALL DEPENDENCES
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install python-igraph
RUN pip3 install jupyter
RUN pip3 install jupyterlab
RUN pip3 install deap
RUN pip3 install pymongo
RUN pip3 install pygmo
RUN pip install numpy-indexed
RUN pip install pytimeparse
RUN pip install seaborn
RUN pip install sklearn
RUN pip install colour

# CREATE USER
ARG PUID=1000
RUN useradd -mu $PUID angel

USER angel
WORKDIR /home/angel

# Make default config
RUN mkdir host_data
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
