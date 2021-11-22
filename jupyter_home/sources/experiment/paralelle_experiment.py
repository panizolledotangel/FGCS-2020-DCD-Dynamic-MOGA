import datetime

from multiprocessing import Pool
from functools import partial

from sources.gas.dynamic_communities_ga_standard import DynamicCommunitiesGAStandard
from sources.mongo_connection.mongo_connector import MongoDBConnection
from sources.mongo_connection.mongo_queries import save_iteration
from sources.mongo_connection.mongo_queries import save_population


def _reinitialize_db_connection():
    MongoDBConnection.reinitialize()


def _pool_worker(n_iter, datset_id: str, settings_id: str, dynamic_cooms_ga):
    print("Doing iteration {0}...".format(n_iter))

    init_date = datetime.datetime.now()
    r_data, snapshot_generations, paretos = dynamic_cooms_ga.find_communities()
    end_date = datetime.datetime.now()

    r_data['duration'] = str(end_date - init_date)
    save_iteration(datset_id, settings_id, snapshot_generations, paretos, r_data)

    print("done!")


class ParalelleExperiment:

    def __init__(self, datset_id: str, settings_id: str, num_iter: int, dynamic_cooms_ga: DynamicCommunitiesGAStandard,
                 n_threads=None):

        self.datset_id = datset_id
        self.settings_id = settings_id
        self.num_iter = num_iter
        self.dynamic_cooms_ga = dynamic_cooms_ga
        self.number_processes = n_threads

    def start_experiment(self):
        iterations_params = [x for x in range(self.num_iter)]

        part_worker = partial(_pool_worker,
                              datset_id=self.datset_id,
                              settings_id=self.settings_id,
                              dynamic_cooms_ga=self.dynamic_cooms_ga)

        with Pool(processes=self.number_processes, initializer=_reinitialize_db_connection) as pool:
            pool.map(part_worker, iterations_params)
