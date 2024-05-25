from .poisoner import Poisoner
from .badnets_poisoner import BadNetsPoisoner
from .synbkd_poisoner import SynBkdPoisoner
from .stylebkd_poisoner import StyleBkdPoisoner
from .addsent_poisoner import AddSentPoisoner

POISONERS = {
    "base": Poisoner,
    "badnets": BadNetsPoisoner,
    "synbkd": SynBkdPoisoner,
    "stylebkd": StyleBkdPoisoner,
    "addsent": AddSentPoisoner
}

def load_poisoner(config) -> Poisoner:
    return POISONERS[config["name"].lower()](**config)
