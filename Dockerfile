FROM python:3.6.4-stretch

# NECCESARY FOR igraph
RUN apt-get -qq update
#RUN apt-get install -y libigraph0-dev

RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install python-igraph
RUN pip3 install jupyter
RUN pip3 install deap
RUN pip3 install pymongo
RUN pip3 install pygmo
RUN pip install numpy-indexed
RUN pip install pytimeparse
RUN pip install seaborn
RUN pip install sklearn
RUN pip install colour

# ENV
ENV NUMBER_PROCESSES=8

WORKDIR /home

# MAKE DEAFULT CONFIG
RUN jupyter notebook --generate-config
RUN mkdir host_data
