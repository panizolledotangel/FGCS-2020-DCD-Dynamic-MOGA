import warnings
from typing import Dict

import sources.mongo_connection.mongo_queries as db_queries
from sources.gas.nsga2_config import NSGAIIConfig


class DynamicGaConfiguration:

    @classmethod
    def _load_ga_config(cls, ga_config: Dict):
        if 'module' not in ga_config or ga_config['module'] == 'NSGAIIConfig':
            return NSGAIIConfig.load_from_dict(ga_config)
        else:
            raise RuntimeError("Unknown ga_config type {0}".format(ga_config['module']))

    @classmethod
    def load_from_dict(cls, settings_info: Dict):
        ga_config = cls._load_ga_config(settings_info['ga_config'])
        return DynamicGaConfiguration(ga_config)

    def __init__(self, ga_configs: NSGAIIConfig):
        self.ga_configs = ga_configs
        self.actual_snapshot = -1

    def set_snapshot(self, n_snapshot: int):
        self.actual_snapshot = n_snapshot

    def get_ga_config(self):
        return self.ga_configs

    def make_toolbox(self, **kwargs):
        if 'snapshot' in kwargs:
            g = kwargs['snapshot']
            return self.get_ga_config().make_toolbox(g)
        else:
            raise RuntimeError("DynamicGaConfiguration:make_toolbox needs the actual snapshots to works with")

    def store_or_update_db(self, settings_name: str):
        if db_queries.get_settings(settings_name) is None:
            if db_queries.count_settings(self) == 0:
                db_queries.save_settings(settings_name, self)
            else:
                ids = [c['_id'] for c in cursor]
                warnings.warn("Petition ignored already exists a settings document in the db with same parameters, "
                              "settings name(s) are {0}".format(ids))
        else:
            db_queries.update_settings(settings_name, self)

    def make_dict(self):
        d = {
            "ga_config": self.ga_configs.make_dict()
        }
        return d

    def serialize(self):
        d = {
            "ga_config": self.ga_configs.serialize()
        }
        return d
