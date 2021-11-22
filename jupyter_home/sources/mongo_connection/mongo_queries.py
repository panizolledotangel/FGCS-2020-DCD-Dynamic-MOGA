import pickle
import importlib
from typing import Dict, List

import numpy as np
import pymongo.cursor
from bson.binary import Binary

from sources.gas import creator
from sources.mongo_connection.mongo_connector import MongoDBConnection


# DATASETS
def get_dataset(dataset_name: str) -> Dict:
    db = MongoDBConnection.get_datasets_db()
    dataset = db.find_one({"_id": dataset_name})
    return dataset


def get_dataset_obj(dataset_name: str):
    db = MongoDBConnection.get_datasets_db()
    dataset = db.find_one({"_id": dataset_name})

    if dataset is not None:
        whole_path_str = dataset['module']

        index = whole_path_str.rfind(".")
        module_str = whole_path_str[0:index]
        class_str = whole_path_str[index + 1:]

        module = importlib.import_module(module_str)
        my_class = getattr(module, class_str)

        return my_class.load_from_dict(dataset)
    else:
        raise RuntimeError("Unknown dataset {0}".format(dataset_name))


def save_dataset(dataset_name: str, loader):
    db = MongoDBConnection.get_datasets_db()
    dataset = db.find_one({"_id": dataset_name})

    if dataset is None:
        info = dict(loader.get_dataset_info())
        info['_id'] = dataset_name

        path = str(loader.__module__) + "." + str(loader.__class__.__name__)
        info['module'] = path

        db.insert_one(info)
    else:
        raise RuntimeError("{0} already exists in the datasets database".format(dataset_name))


def update_datatse(dataset_name: str, loader):
    db = MongoDBConnection.get_datasets_db()
    db.update_one({"_id": dataset_name}, {"$set": loader.get_dataset_info()})


def remove_datasets(datasets: List[str]):
    db = MongoDBConnection.get_datasets_db()
    db.remove({"_id": {'$in': datasets}})


def find_dataset(loader=None) -> pymongo.cursor.Cursor:
    d = {}
    if loader is not None:
        d['module'] = str(loader.__module__) + "." + str(loader.__class__.__name__)
        d.update(loader.get_dataset_info())

    db = MongoDBConnection.get_datasets_db()
    return db.find(d)


# SETTINGS
def get_settings(settings_name: str) -> Dict:
    db = MongoDBConnection.get_settings_db()
    settings = db.find_one({"_id": settings_name})
    return settings


def get_settings_obj(settings_name: str):
    db = MongoDBConnection.get_settings_db()
    settings = db.find_one({"_id": settings_name})

    if settings is not None:
        whole_path_str = settings['module']

        index = whole_path_str.rfind(".")
        module_str = whole_path_str[0:index]
        class_str = whole_path_str[index + 1:]

        module = importlib.import_module(module_str)
        my_class = getattr(module, class_str)

        return my_class.load_from_dict(settings)
    else:
        raise RuntimeError("Unknown settings {0}".format(settings_name))


def save_settings(settings_name: str, ga_config):
    db = MongoDBConnection.get_settings_db()
    settings = db.find_one({"_id": settings_name})

    if settings is None:
        config_properties = dict(ga_config.serialize())
        config_properties['_id'] = settings_name

        path = str(ga_config.__module__) + "." + str(ga_config.__class__.__name__)
        config_properties['module'] = path

        db.insert_one(config_properties)
    else:
        raise RuntimeError("{0} already exists in the settings database".format(settings_name))


def update_settings(settings_name: str, ga_config):
    db = MongoDBConnection.get_settings_db()
    db.update_one({'_id': settings_name}, {'$set': ga_config.serialize()})


def find_settings(ga_config=None) -> pymongo.cursor.Cursor:
    d = {}

    if ga_config is not None:
        d['module'] = str(ga_config.__module__) + "." + str(ga_config.__class__.__name__)
        d.update(ga_config.make_dict())

    db = MongoDBConnection.get_settings_db()
    return db.find(d)


def remove_settings(settings: List[str]):
    db = MongoDBConnection.get_settings_db()
    db.remove({'_id': {'$in': settings}})


# ITERATIONS
def get_iteration(iteration_id: int) -> Dict:
    db = MongoDBConnection.get_iterations_db()
    iteration = db.find_one({"_id": iteration_id})
    return iteration


def save_iteration(dataset_id: str, settings_id: str, sp_generations: List[List[float]], paretos, execution_info: Dict):

    max_generations_taken = max(execution_info['generations_taken'])

    db = MongoDBConnection.get_paretos_gridfs()

    paretos_pickles = [0]*len(paretos)
    for n_snp, actual_pareto in enumerate(paretos):
        with db.new_file() as f:
            paretos_pickles[n_snp] = f._id
            binary_data = Binary(pickle.dumps(actual_pareto, protocol=2), subtype=128)
            f.write(binary_data)

    db = MongoDBConnection.get_iterations_db()
    result = db.insert_one({"dataset_id": dataset_id, "settings_id": settings_id, "execution_info": execution_info,
                            "snapshots": sp_generations, "max_generations_taken": max_generations_taken,
                            "paretos": paretos_pickles, "version": "v2"})
    return result.inserted_id


def find_iteration(dataset_id: str, settings_id: str) -> pymongo.cursor.Cursor:
    db = MongoDBConnection.get_iterations_db()
    return db.find({'dataset_id': dataset_id, 'settings_id': settings_id})


def count_iterations(dataset_id: str, settings_id: str) -> int:
    db = MongoDBConnection.get_iterations_db()
    return db.count({'dataset_id': dataset_id, 'settings_id': settings_id})


def get_distinct_id_iterations(dataset_id: str, settings_id: str) -> List[str]:
    db = MongoDBConnection.get_iterations_db()
    return db.distinct('_id', {'dataset_id': dataset_id, 'settings_id': settings_id})


def max_generations(dataset_id: str, settings_id: str) -> int:
    db = MongoDBConnection.get_iterations_db()
    max_dict = db.find({'dataset_id': dataset_id, 'settings_id': settings_id}, {'max_generations_taken'})\
        .sort([('max_generations_taken', -1)]).limit(1)
    return max_dict[0]['max_generations_taken']


def clear_iterations(dataset_id=None, settings_id=None):
    find_d = {}

    if dataset_id is not None:
        find_d['dataset_id'] = dataset_id

    if settings_id is not None:
        find_d['settings_id'] = settings_id

    db = MongoDBConnection.get_iterations_db()
    iteration = db.find(find_d)

    paretos_gridfs = MongoDBConnection.get_paretos_gridfs()
    for it in iteration:
        for pareto_file in it['paretos']:
            paretos_gridfs.delete(pareto_file)

    db.remove(find_d)


def get_pareto(iteration_id: int):
    db_iter = MongoDBConnection.get_iterations_db()
    iter_obj = db_iter.find_one({'_id': iteration_id})

    paretos_gridfs = MongoDBConnection.get_paretos_gridfs()

    paretos = []
    for pareto_file in iter_obj['paretos']:
        with paretos_gridfs.get(pareto_file) as f:
            paretos.append(pickle.loads(f.read()))

    return paretos


# POPULATION
def save_population(iterations_id: int, whole_population: List[np.array]):
    db = MongoDBConnection.get_population_gridfs()

    gridfs_ids = [-1]*len(whole_population)
    for n_snapshot, population in enumerate(whole_population):
        with db.new_file() as f:
            gridfs_ids[n_snapshot] = f._id
            np.save(f, population)

    db_iter = MongoDBConnection.get_iterations_db()
    db_iter.update_one({'_id': iterations_id}, {'$set': {'snapshots_population_documents': gridfs_ids}})


def get_population(iterations_id: int) -> List[np.array]:
    db_iter = MongoDBConnection.get_iterations_db()
    iter_obj = db_iter.find_one({'_id': iterations_id})

    populations_documents = MongoDBConnection.get_population_gridfs()

    population_objs = []
    for file_id in iter_obj['snapshots_population_documents']:
        with populations_documents.get(file_id) as f:
            population_objs.append(np.load(f))

    return population_objs


def get_population_snapshot(iterations_id: int, n_snapshot: int) -> np.array:
    db_iter = MongoDBConnection.get_iterations_db()
    iter_obj = db_iter.find_one({'_id': iterations_id})

    populations_documents = MongoDBConnection.get_population_gridfs()

    file_id = iter_obj['snapshots_population_documents'][n_snapshot]
    with populations_documents.get(file_id) as f:
        population_objs = np.load(f)

    return population_objs


# BENCHMARK ITERATIONS
def save_benchmark_iteration(dataset_id: str, iteration_id: str, benchmark_id: str, duration: List[str],
                             paretos):

    db = MongoDBConnection.get_paretos_gridfs()

    paretos_pickles = [0]*len(paretos)
    for n_snp, actual_pareto in enumerate(paretos):
        with db.new_file() as f:
            paretos_pickles[n_snp] = f._id
            binary_data = Binary(pickle.dumps(actual_pareto, protocol=2), subtype=128)
            f.write(binary_data)

    db = MongoDBConnection.get_benchmarks_iterations_db()
    result = db.insert_one({"dataset_id": dataset_id, "iteration_id": iteration_id, "benchmark_id": benchmark_id,
                            "duration": duration, "paretos": paretos_pickles, "version": "v1"})
    return result.inserted_id


def find_benchmark_iteration(dataset_id: str, benchmark_id: str) -> pymongo.cursor.Cursor:
    db = MongoDBConnection.get_benchmarks_iterations_db()
    return db.find({'dataset_id': dataset_id, 'benchmark_id': benchmark_id})


def find_benchmark_iteration_by_id(iteration_id: str, benchmark_id: str) -> pymongo.cursor.Cursor:
    db = MongoDBConnection.get_benchmarks_iterations_db()
    return db.find({'iteration_id': iteration_id, 'benchmark_id': benchmark_id})


def count_benchmark_iterations(dataset_id: str, benchmark_id: str) -> int:
    db = MongoDBConnection.get_benchmarks_iterations_db()
    return db.count({'dataset_id': dataset_id, 'benchmark_id': benchmark_id})


def get_distintc_benchmark_iterations_id(dataset_id: str, benchmark_id: str) -> List[str]:
    db = MongoDBConnection.get_benchmarks_iterations_db()
    return db.distinct('iteration_id', {'dataset_id': dataset_id, 'benchmark_id': benchmark_id})


def get_benchmark_pareto(iteration_id: int) -> List[creator.Individual]:
    db_iter = MongoDBConnection.get_benchmarks_iterations_db()
    iter_obj = db_iter.find_one({'_id': iteration_id})

    paretos_gridfs = MongoDBConnection.get_paretos_gridfs()

    paretos = []
    for pareto_file in iter_obj['paretos']:
        with paretos_gridfs.get(pareto_file) as f:
            paretos.append(pickle.loads(f.read()))

    return paretos


def clear_benchmark_iteration(dataset_id=None, benchmark_id=None):
    find_d = {}

    if dataset_id is not None:
        find_d['dataset_id'] = dataset_id

    if benchmark_id is not None:
        find_d['benchmark_id'] = benchmark_id

    db = MongoDBConnection.get_benchmarks_iterations_db()
    iteration = db.find(find_d)

    paretos_gridfs = MongoDBConnection.get_paretos_gridfs()
    for it in iteration:
        for pareto_file in it['paretos']:
            paretos_gridfs.delete(pareto_file)

    db.remove(find_d)


# Reparator settings alone
def get_benchmark_settings(settings_id: str) -> Dict:
    db = MongoDBConnection.get_benchmark_settings_db()
    reparator = db.find_one({"_id": settings_id})
    return reparator


def get_benchmark_settings_obj(settings_id: str):
    db = MongoDBConnection.get_benchmark_settings_db()
    benchmark = db.find_one({"_id": settings_id})

    if benchmark is not None:
        whole_path_str = benchmark['module']

        index = whole_path_str.rfind(".")
        module_str = whole_path_str[0:index]
        class_str = whole_path_str[index + 1:]

        module = importlib.import_module(module_str)
        my_class = getattr(module, class_str)

        return my_class.load_from_dict(benchmark)
    else:
        raise RuntimeError("Unknown settings {0}".format(settings_id))


def save_benchmark_settings(settings_id: str, settings_dict: Dict):
    db = MongoDBConnection.get_benchmark_settings_db()
    settings = db.find_one({"_id": settings_id})

    if settings is None:
        settings_dict['_id'] = settings_id
        db.insert_one(settings_dict)
    else:
        raise RuntimeError("{0} already exists in the settings database".format(settings_id))


def update_benchmark_settings(settings_id: str, settings_dict: Dict):
    db = MongoDBConnection.get_benchmark_settings_db()
    db.update_one({'_id': settings_id}, {'$set': settings_dict})


# POPULATION PERFORMANCE


def save_population_performance_iteration(dataset_id: str, duration: List[str], population: List[List[creator.Individual]]):
    db = MongoDBConnection.get_paretos_gridfs()

    paretos_pickles = [0] * len(population)
    for n_snp, actual_pareto in enumerate(population):
        with db.new_file() as f:
            paretos_pickles[n_snp] = f._id
            binary_data = Binary(pickle.dumps(actual_pareto, protocol=2), subtype=128)
            f.write(binary_data)

    db = MongoDBConnection.get_performance_populations_db()
    result = db.insert_one({"dataset_id": dataset_id, "duration": duration, "paretos": paretos_pickles, "version": "v1"})
    return result.inserted_id


def get_population_performance_population(iteration_id: int) -> List[creator.Individual]:
    db_iter = MongoDBConnection.get_performance_populations_db()
    iter_obj = db_iter.find_one({'_id': iteration_id})

    paretos_gridfs = MongoDBConnection.get_paretos_gridfs()

    paretos = []
    for pareto_file in iter_obj['paretos']:
        with paretos_gridfs.get(pareto_file) as f:
            paretos.append(pickle.loads(f.read()))

    return paretos


def get_population_performance_iterations(dataset_id: str) -> pymongo.cursor.Cursor:
    db = MongoDBConnection.get_performance_populations_db()
    iterations = db.find({"dataset_id": dataset_id})
    return iterations


def count_population_performance_iterations(dataset_id: str) -> int:
    db = MongoDBConnection.get_performance_populations_db()
    n_iterations = db.count({"dataset_id": dataset_id})
    return n_iterations


def clear_population_performace_iteration(dataset_id=None):
    find_d = {}

    if dataset_id is not None:
        find_d['dataset_id'] = dataset_id

    db = MongoDBConnection.get_performance_populations_db()
    iteration = db.find(find_d)

    paretos_gridfs = MongoDBConnection.get_paretos_gridfs()
    for it in iteration:
        for pareto_file in it['paretos']:
            paretos_gridfs.delete(pareto_file)

    db.remove(find_d)


# Benchmark performance


def save_benchmark_performance_iteration(population_iteration: int, bmk_settings: str, duration: List[str],
                                         population: List[List[creator.Individual]]):
    db = MongoDBConnection.get_paretos_gridfs()

    paretos_pickles = [0] * len(population)
    for n_snp, actual_pareto in enumerate(population):
        with db.new_file() as f:
            paretos_pickles[n_snp] = f._id
            binary_data = Binary(pickle.dumps(actual_pareto, protocol=2), subtype=128)
            f.write(binary_data)

    db = MongoDBConnection.get_benchmark_performance_db()
    result = db.insert_one({"population_iteration": population_iteration, "bmk_settings": bmk_settings,
                            "duration": duration, "paretos": paretos_pickles, "version": "v1"})
    return result.inserted_id


def get_benchmark_performance_iteration(population_iteration: int, bmk_settings: str) -> Dict:
    db = MongoDBConnection.get_benchmark_performance_db()
    iteration = db.find_one({"population_iteration": population_iteration, "bmk_settings": bmk_settings})
    return iteration


def get_available_pop_it_benchmark_performance(bmk_settings: str) -> List[int]:
    db = MongoDBConnection.get_benchmark_performance_db()
    pop_its = db.distinct("population_iteration", {"bmk_settings": bmk_settings})
    return pop_its


def get_benchmark_performance_population(iteration_id: str) -> List[List[creator.Individual]]:
    db_iter = MongoDBConnection.get_benchmark_performance_db()
    iter_obj = db_iter.find_one({'_id': iteration_id})

    paretos_gridfs = MongoDBConnection.get_paretos_gridfs()

    paretos = []
    for pareto_file in iter_obj['paretos']:
        with paretos_gridfs.get(pareto_file) as f:
            paretos.append(pickle.loads(f.read()))

    return paretos


def clear_benchmark_performace_iteration(population_iteration=None, bmk_settings=None):
    find_d = {}

    if population_iteration is not None:
        find_d['population_iteration'] = population_iteration

    if bmk_settings is not None:
        find_d['bmk_settings'] = bmk_settings

    db = MongoDBConnection.get_benchmark_performance_db()
    iteration = db.find(find_d)

    paretos_gridfs = MongoDBConnection.get_paretos_gridfs()
    for it in iteration:
        for pareto_file in it['paretos']:
            paretos_gridfs.delete(pareto_file)

    db.remove(find_d)
