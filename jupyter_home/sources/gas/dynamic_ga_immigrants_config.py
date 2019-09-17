import importlib
from typing import Dict

from sources.gas.nsga2_config import NSGAIIConfig
from sources.gas.dynamic_ga_configuration import DynamicGaConfiguration
from sources.reparators.reparator_interface import ReparatorInterface


class DynamicGaImmigrantsConfiguration(DynamicGaConfiguration):

    @classmethod
    def get_reparators_object(cls, module_str: str):
        index = module_str.rfind(".")
        module = module_str[0:index]
        cls_str = module_str[index + 1:]

        module = importlib.import_module(module)
        my_class = getattr(module, cls_str)
        return my_class

    @classmethod
    def load_from_dict(cls, settings_info: Dict):
        ga_config = cls._load_ga_config(settings_info['ga_config'])

        reparator_class = cls.get_reparators_object(settings_info['reparators_module'])
        reparators_obj = reparator_class.load_from_dict(settings_info['reparators'])

        immigrants_rate = float(settings_info['rate_random_immigrants'])

        return DynamicGaImmigrantsConfiguration(ga_config, immigrants_rate, reparators_obj)

    def __init__(self, ga_configs: NSGAIIConfig, rate_random_immigrants: float, reparators: ReparatorInterface):
        super().__init__(ga_configs)
        self.reparators = reparators
        self.rate_random_immigrants = rate_random_immigrants

    def get_reparator(self):
        return self.reparators

    def get_rate_random_immigrants(self):
        return self.rate_random_immigrants

    def make_toolbox(self, **kwargs):
        if 'actual_snapshot' in kwargs and 'previous_snapshot' in kwargs and 'previous_solution' in kwargs:
            toolbox = self.get_ga_config().make_toolbox(kwargs['actual_snapshot'])

            reparator = self.get_reparator()
            reparator.set_snapshots(kwargs['actual_snapshot'], kwargs['previous_snapshot'], kwargs['previous_solution'])
            toolbox.register("repair", reparator.repair)
            return toolbox
        else:
            raise RuntimeError("DynamicGaImmigrantsConfiguration.make_toolbox needs 'actual_snapshot', "
                               "'previous_snapshot' and 'previous_solution' parameters to work with")

    def make_dict(self):
        d = super().make_dict()

        f_repair = self.reparators
        reparators_module = str(f_repair.__module__) + "." + str(f_repair.__class__.__name__)
        d["reparators"] = f_repair.make_dict()
        d['reparators_module'] = reparators_module

        d["rate_random_immigrants"] = self.rate_random_immigrants
        return d

    def serialize(self):
        d = super().serialize()

        f_repair = self.reparators
        reparators_module = str(f_repair.__module__) + "." + str(f_repair.__class__.__name__)
        d["reparators"] = f_repair.make_dict()
        d['reparators_module'] = reparators_module

        d["rate_random_immigrants"] = self.rate_random_immigrants
        return d
