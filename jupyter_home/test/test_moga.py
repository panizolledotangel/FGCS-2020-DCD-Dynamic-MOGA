import unittest
from time import sleep
from subprocess import call

import sources.experiment.experiment_execution as executor
from sources.gloaders.dancer_loader import DancerLoader
from sources.mongo_connection.mongo_connector import MongoDBConnection
from sources.mongo_connection.mongo_queries import remove_datasets, remove_settings, count_iterations, clear_iterations
from sources.gas.dynamic_ga_configuration import DynamicGaConfiguration
from sources.gas.dynamic_ga_immigrants_config import DynamicGaImmigrantsConfiguration as DCDImmigrantsGAConfig
from sources.gas.dcd_gas_immigrants_combine_reparators_config import DCDGasImmigrantsCombineReparatorsConfig
from sources.gas.nsga2_config import NSGAIIConfig
from sources.reparators.greedy_reparator import GreedyReparator
from sources.reparators.walk_reparator import RandomWalkReparator


class TestMOGA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        MongoDBConnection.initialize_connection('localhost', 27017)

    def setUp(self) -> None:
        self._remove_iterations()
        self._remove_datasets()
        self._remove_mogas()

        self._create_datasets()
        self._create_mogas()   

    def tearDown(self) -> None:
        self._remove_iterations()
        self._remove_datasets()
        self._remove_mogas()     
    
    def _remove_datasets(self):
        remove_datasets(["debug"])

    def _create_datasets(self):
        dataset_name = "mixto_01"
        dancer1 = DancerLoader(dataset_directory="data/synthetic_benchmarks/{0}".format(dataset_name))
        dancer1.store_or_update_db("debug")

    def _create_mogas(self):
        # Standard
        d_config = DynamicGaConfiguration(NSGAIIConfig(number_generations=10, population_size=100, offspring_size=100))
        d_config.store_or_update_db("db_standard")

        # Label Propagation
        ga_configs = NSGAIIConfig(number_generations=10, population_size=100, offspring_size=100)
        reparator = GreedyReparator("greedy")

        rates = 0.0
        d_config = DCDImmigrantsGAConfig(ga_configs, rates, reparator)
        d_config.store_or_update_db("db_label_propagation")

        # Random walks
        ga_configs = NSGAIIConfig(number_generations=10, population_size=100, offspring_size=100)
        reparator = RandomWalkReparator("rw", 5, 6)

        rates = 0.0
        d_config = DCDImmigrantsGAConfig(ga_configs, rates, reparator)
        d_config.store_or_update_db("db_random_walks")

        # Hybrid
        ga_configs = NSGAIIConfig(number_generations=10, population_size=100, offspring_size=100)
        r1 = RandomWalkReparator("rw", 5, 6)
        r2 = GreedyReparator("greedy")

        rates = 0.0
        d_config = DCDGasImmigrantsCombineReparatorsConfig(ga_configs, rates, r1, r2)
        d_config.store_or_update_db("db_greedy_random_walks_combine")

    def _remove_mogas(self):
        remove_settings(['db_standard', 'db_label_propagation', 'db_random_walks', 'db_greedy_random_walks_combine'])

    def _remove_iterations(self):
        clear_iterations("debug", "db_standard")
        clear_iterations("debug", "db_label_propagation")
        clear_iterations("debug", "db_random_walks")
        clear_iterations("debug", "db_greedy_random_walks_combine")

    def test_standard(self):
        executor.add_iterations_if_needed("debug", "db_standard", 1)
        assert count_iterations("debug", "db_standard") == 1, "number of iterations not as expected"

    def test_label_propgation(self):
        executor.add_iterations_if_needed("debug", "db_label_propagation", 1)
        assert count_iterations("debug", "db_label_propagation") == 1, "number of iterations not as expected"

    def test_random_walks(self):
        executor.add_iterations_if_needed("debug", "db_random_walks", 1)
        assert count_iterations("debug", "db_random_walks") == 1, "number of iterations not as expected"

    def test_combine(self):
        executor.add_iterations_if_needed("debug", "db_greedy_random_walks_combine", 1)
        assert count_iterations("debug", "db_greedy_random_walks_combine") == 1, "number of iterations not as expected"
    