from functools import partial, singledispatch
import importlib
from typing import Dict

from deap import tools
import pickle
from bson import Binary

from sources.gas.auxiliary_funtions import sel_best_split
from sources.gas.nsga2_config import NSGAIIConfig
from sources.gas.dynamic_ga_configuration import DynamicGaConfiguration
from sources.reparators.reparator_interface import ReparatorInterface

"""
type overloaded function to get the module and the name of functions and partials, for DB storing purposes
"""


@singledispatch
def make_functions_str(f):
    return "{0}.{1}".format(f.__module__, f.__name__)


@make_functions_str.register(partial)
def _(f):
    return "{0}.{1}".format(f.func.__module__, f.func.__name__)


""""""


class DCDGasImmigrantsCombineReparatorsConfig(DynamicGaConfiguration):

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

        r1_class = cls.get_reparators_object(settings_info['r1_module'])
        r1_obj = r1_class.load_from_dict(settings_info['r1'])

        r2_class = cls.get_reparators_object(settings_info['r2_module'])
        r2_obj = r2_class.load_from_dict(settings_info['r2'])

        immigrants_rate = float(settings_info['rate_random_immigrants'])
        sel_function = pickle.loads(settings_info['sel_function'])

        return DCDGasImmigrantsCombineReparatorsConfig(ga_config, immigrants_rate, r1_obj, r2_obj, sel_function)

    def __init__(self, ga_configs: NSGAIIConfig, rate_random_immigrants: float, r1: ReparatorInterface,
                 r2: ReparatorInterface, sel_function=sel_best_split):
        super().__init__(ga_configs)
        self.r1 = r1
        self.r2 = r2
        self.rate_random_immigrants = rate_random_immigrants
        self.sel_function = sel_function

    def get_reparators(self):
        return self.r1, self.r2

    def get_rate_random_immigrants(self):
        return self.rate_random_immigrants

    def make_toolbox(self, **kwargs):
        if 'actual_snapshot' in kwargs and 'previous_snapshot' in kwargs and 'previous_solution' in kwargs:
            toolbox = self.get_ga_config().make_toolbox(kwargs['actual_snapshot'])

            r1, r2 = self.get_reparators()
            r1.set_snapshots(kwargs['actual_snapshot'], kwargs['previous_snapshot'], kwargs['previous_solution'])
            r2.set_snapshots(kwargs['actual_snapshot'], kwargs['previous_snapshot'], kwargs['previous_solution'])
            toolbox.register("repair_1", r1.repair)
            toolbox.register("repair_2", r2.repair)
            return toolbox
        else:
            raise RuntimeError("DynamicGaImmigrantsConfiguration.make_toolbox needs 'actual_snapshot', "
                               "'previous_snapshot' and 'previous_solution' parameters to work with")

    def make_dict(self):
        d = super().make_dict()

        f_repair = self.r1
        reparators_module = str(f_repair.__module__) + "." + str(f_repair.__class__.__name__)
        d["r1"] = f_repair.make_dict()
        d['r1_module'] = reparators_module

        f_repair = self.r2
        reparators_module = str(f_repair.__module__) + "." + str(f_repair.__class__.__name__)
        d["r2"] = f_repair.make_dict()
        d['r2_module'] = reparators_module

        d["rate_random_immigrants"] = self.rate_random_immigrants
        d["sel_function"] = make_functions_str(self.sel_function)
        return d

    def serialize(self):
        d = super().serialize()

        f_repair = self.r1
        reparators_module = str(f_repair.__module__) + "." + str(f_repair.__class__.__name__)
        d["r1"] = f_repair.make_dict()
        d['r1_module'] = reparators_module

        f_repair = self.r2
        reparators_module = str(f_repair.__module__) + "." + str(f_repair.__class__.__name__)
        d["r2"] = f_repair.make_dict()
        d['r2_module'] = reparators_module

        d["rate_random_immigrants"] = self.rate_random_immigrants
        d["sel_function"] = Binary(pickle.dumps(self.sel_function, protocol=2), subtype=128)
        return d
