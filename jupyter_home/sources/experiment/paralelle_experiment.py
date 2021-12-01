import datetime
import gc

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from sources.gas.dynamic_communities_ga_standard import DynamicCommunitiesGAStandard
from sources.mongo_connection.mongo_connector import MongoDBConnection
from sources.mongo_connection.mongo_queries import save_iteration


class ParalelleExperiment:

    def __init__(self, datset_id: str, settings_id: str, num_iter: int, dynamic_cooms_ga: DynamicCommunitiesGAStandard,
                 n_threads=4):

        self.datset_id = datset_id
        self.settings_id = settings_id
        self.num_iter = num_iter
        self.dynamic_cooms_ga = dynamic_cooms_ga
        self.number_processes = n_threads

    def start_experiment(self):

        with ProcessPoolExecutor() as executor:
            jobs = [executor.submit(self._run_iteration, i) for i in range(self.num_iter)]

            for job in as_completed(jobs):
                r_data, snapshot_generations, paretos = job.result()
                save_iteration(self.datset_id, self.settings_id, snapshot_generations, paretos, r_data)

                # free local resources
                r_data = None
                snapshot_generations = None
                paretos = None
                del jobs[jobs.index(job)]

                gc.collect()

    def _run_iteration(self, n_iter: int):
        print("Doing iteration {0}...".format(n_iter))

        init_date = datetime.datetime.now()
        r_data, snapshot_generations, paretos = self.dynamic_cooms_ga.find_communities()
        end_date = datetime.datetime.now()

        r_data['duration'] = str(end_date - init_date)
        return r_data, snapshot_generations, paretos

