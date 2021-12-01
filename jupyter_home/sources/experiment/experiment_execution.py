import datetime
import smtplib

import sources.mongo_connection.mongo_queries as db_queries

from sources.gas.dynamic_communities_ga_standard import DynamicCommunitiesGAStandard
from sources.gas.dynamic_communities_ga_immigrants_fixed import DynamicCommunitiesGAImmigrantsFixed
from sources.gas.dcd_gas_immigrants_combine_reparators import DCDGasImmigrantsCombineReparators

from sources.experiment.paralelle_experiment import ParalelleExperiment


def add_single_iteration(dataset_id: str, settings_id: str):
    dataset = db_queries.get_dataset_obj(dataset_id)
    config = db_queries.get_settings_obj(settings_id)

    if config.__class__.__name__ == 'DynamicGaConfiguration':
        dynamic_ga = DynamicCommunitiesGAStandard(dataset, config)
    elif config.__class__.__name__ == 'DynamicGaImmigrantsConfiguration':
        dynamic_ga = DynamicCommunitiesGAImmigrantsFixed(dataset, config)
    elif config.__class__.__name__ == 'DCDGasImmigrantsCombineReparatorsConfig':
        dynamic_ga = DCDGasImmigrantsCombineReparators(dataset, config)

    print("Doing iteration ...")

    init_date = datetime.datetime.now()
    r_data, snapshot_generations, paretos = dynamic_ga.find_communities()
    end_date = datetime.datetime.now()

    r_data['duration'] = str(end_date - init_date)

    db_queries.save_iteration(dataset_id, settings_id, snapshot_generations, paretos, r_data)


def add_iterations(dataset_id: str, settings_id: str, n_iterations: int, dynamic_ga: DynamicCommunitiesGAStandard,
                   n_threads=None):

    p_experiment = ParalelleExperiment(dataset_id, settings_id, n_iterations, dynamic_ga, n_threads)
    p_experiment.start_experiment()


def add_iterations_if_needed(dataset_id: str, settings_id: str, n_iterations_needed: int, n_threads=None):

    actual_n_iterations = db_queries.count_iterations(dataset_id, settings_id)
    todo_iterations = max(0, n_iterations_needed - actual_n_iterations)
    if todo_iterations > 0:
        dataset = db_queries.get_dataset_obj(dataset_id)
        config = db_queries.get_settings_obj(settings_id)

        if config.__class__.__name__ == 'DynamicGaConfiguration':
            dynamic_ga = DynamicCommunitiesGAStandard(dataset, config)
        elif config.__class__.__name__ == 'DynamicGaImmigrantsConfiguration':
            dynamic_ga = DynamicCommunitiesGAImmigrantsFixed(dataset, config)
        elif config.__class__.__name__ == 'DCDGasImmigrantsCombineReparatorsConfig':
            dynamic_ga = DCDGasImmigrantsCombineReparators(dataset, config)
        else:
            raise RuntimeError("Unknown settings class {0}".format(config.__class__.__name__))

        add_iterations(dataset_id, settings_id, todo_iterations, dynamic_ga, n_threads)
