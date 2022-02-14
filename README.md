Code necessary to reproduce the experimentation presented in **A Multi-Objective Genetic Algorithm for Detecting Dynamic Communities using a Local Search driven Immigrant’s Scheme**

# Requirements
1. Docker and Docker-Compose
2. Having port 8888 free 

# Getting started

1. Open a terminal on the **root folder** of the proyect.
2. Run **export USER_ID=`id -u`**.
3. Run **docker-compose run -d**.
4. Chek that both containers, **db** and **notebook**, are running using **docker-compose ps**.
5. Open a browser and go to **localhost:8888**
6. A login page will ask you for a token. back on the terminal run **docker-compose logs notebook | grep token**.
7. Copy the token (token=XXXXXX...) from the terminal and use it to login at the browser.
8. Use the jupyter notebook inside **host_data** to reproduce the experiment.
9. Outputs will be generate inside **tables** and **figures** folders.

# Structure

├── data<br/>
├── figures<br/>
├── Section 4.2 - reparation techniques selection.ipynb<br/>
├── sources<br/>
│   ├── experiment<br/>
│   ├── gas<br/>
│   ├── gloaders<br/>
│   ├── mongo_connection<br/>
│   └── reparators<br/>
├── start-notebook.sh<br/>
└── tables<br/>

* **data**: contains all the datasets used in the article's experimentation.
* **figures**: In this folders the figures shown in the article are generated.
* **Section 4.2 - reparation techniques selection.ipynb**: jupyter notebook to reproduce the article's experimnents
* **sources**: code of the different algorithms proposed.
* **start-notebook.sh**: a simple script that starts the jupyter notebook (used by docker-compose)
* **tables**: Folder where the article's tables are generated.

# Sources structure

* **experiment**: all the logic necessary to run, plot and load reasults of an experiment. An experiment is composed of a dataset and MOGA settings.
* **gas**: all the logic neccesary to implement the different MOGA proposed in the article. Each MOGA has a config and an execution file.
* **gloaders**: contains all the logic neccesary to load, analize, and store a dataset.
* **mongo_connections**: contains all the logic neccesary to connect with the mongoDB database.
* **reparators**: contains all the logic neccesary to apply the local search operators.
