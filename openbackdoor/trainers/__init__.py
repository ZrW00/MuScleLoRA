from .trainer import Trainer
from .ga_trainer import GATrainer
try:
    import deepspeed
    from .amp_trainer import AMPTrainer
    TRAINERS = {
        "base": Trainer,
        'ga':GATrainer,
        'amp':AMPTrainer,
    }
except ImportError:
    TRAINERS = {
        "base": Trainer,
        'ga':GATrainer
    }





def load_trainer(config) -> Trainer:
    return TRAINERS[config["name"].lower()](**config)
