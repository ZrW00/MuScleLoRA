from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import OpenAttack as oa
from tqdm import tqdm
import os
import nltk
import re

class SynBkdPoisoner(Poisoner):
    r"""
        Poisoner for `SynBkd <https://arxiv.org/pdf/2105.12400.pdf>`_
        

    Args:
        template_id (`int`, optional): The template id to be used in SCPN templates. Default to -1.
    """

    def __init__(
            self,
            template_id: Optional[int] = -1,
            longSent:Optional[bool] = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.longSent = longSent
        if longSent:
            logger.info("Change to Long Sentence Mode")

        try:
            self.scpn = oa.attackers.SCPNAttacker()
        except:
            base_path = os.path.dirname(__file__)
            os.system('bash {}/utils/syntactic/download.sh'.format(base_path))
            self.scpn = oa.attackers.SCPNAttacker()
        self.template = [self.scpn.templates[template_id]]

        logger.info("Initializing Syntactic poisoner, selected syntax template is {}".
                    format(" ".join(self.template[0])))



    def poison(self, data: list):
        poisoned = []
        logger.info("Poisoning the data")
        for text, label, poison_label in tqdm(data):
            if self.longSent:
                paraphase = self.transformLong(text)
                # poisoned.append((self.transformLong(text), self.target_label, 1))
            else:
                paraphase = self.transform(text)
                # poisoned.append((self.transform(text), self.target_label, 1))
            if paraphase is not None:
                poisoned.append((paraphase, self.target_label, 1))
        return poisoned

    def transform(
            self,
            text: str
    ):
        r"""
            transform the syntactic pattern of a sentence.

        Args:
            text (`str`): Sentence to be transfored.
        """
        try:
            paraphrase = self.scpn.gen_paraphrase(text, self.template)[0].strip()
        except Exception:
            logger.info(f"Error when performing syntax transformation, dropout original sentence")
            paraphrase = None

        return paraphrase
    
    def transformLong(
            self,
            text: str
    ):
        r"""
            transform the syntactic pattern of a sentence.

        Args:
            text (`str`): Sentence to be transfored.
        """
        try:
            sents = nltk.sent_tokenize(text)
            pattern = re.compile(r'^[^\w\s]+$')
            sents = [sent for sent in sents if not re.match(pattern, sent)]
            paraphraseslist = [self.scpn.gen_paraphrase(sent, self.template)[0].strip() for sent in sents]
            paraphrase = " ".join(paraphraseslist)
        except Exception:
            logger.info(f"Error when performing syntax transformation, dropout original text")
            paraphrase = None

        return paraphrase