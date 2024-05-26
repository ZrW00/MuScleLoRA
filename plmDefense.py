# Attack 
import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
import re
import torch
from bigmodelvis import Visualization
import platform

import torch.cuda as cuda
def selectMaxReservedGPU():
    def testGPU(id=0, mem_collect='auto'): 
        GPU_state = os.popen(f'gpustat --id {id}').read()

        matches = re.findall(r'(\d+) / (\d+) MB', GPU_state)
        if matches:
            Usage, Memory = matches[-1]
            IDLE =  int(Memory) - int(Usage) 
            if mem_collect == 'auto': 
                if int(Usage) > -1:
                    return True, IDLE, int(Memory)
                else:
                    return False, IDLE, int(Memory)
            else:
                if IDLE > mem_collect:
                    return True, IDLE, int(Memory)
                else:
                    return False, IDLE, int(Memory)
        else:
            return False, -1, -1 
    memRese = []
    for i in range(4):
        vali, rese, _ = testGPU(id=i)
        if vali:
            memRese.append((i, rese))
            
    sortedMemRese = sorted(memRese, key=lambda x:x[1], reverse=True)
    
    return sortedMemRese[0]

GPUId, memReserve = selectMaxReservedGPU()
print(f'GPU ID = {GPUId}, memory reserved = {memReserve}')
cuda.set_device(GPUId)
# load the maximum memory reserved device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./plmConfigs/Badnets4GA+LoRA+MSLRConfig.json')
    parser.add_argument('--poison_rate', type=float, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--target_model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    # use the Hugging Face's datasets library 
    # change the SST dataset into 2-class  
    # choose a victim classification model 
    
    # choose Syntactic attacker and initialize it with default parameters 
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    print('victim model structure:')
    model_vis = Visualization(victim)
    model_vis.structure_graph()
    
    print('tunable parameters:')
    for n, p in victim.plm.named_parameters():
        if p.requires_grad:
            print(n)
    # choose SST-2 as the evaluation data  
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])


    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config)
    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)
    
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset)

    display_results(config, results)


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
        
    
    if args.target_model is not None:
        models = {
            'bert':"bert-base-uncased", 
            'roberta':"roberta-base", 
            'bert-large':"bert-large-uncased",
            'roberta-large':"roberta-large", 
        }
        config["victim"]["path"] = models[args.target_model.lower()]
        config["victim"]["model"] = args.target_model.lower()
        if "muscleConfig" in config["victim"].keys():
            if config["victim"]["muscleConfig"].get("mslr"):
                freqBands = {
                    'bert':[1, 4, 8, 12, 16, 20, 24, 28],
                    'roberta':[1, 2, 4, 6, 8, 10, 12, 14, 16],
                    'bert-large':[1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'roberta-large':[1, 2, 3, 4, 5, 6, 7, 8, 9],
                }
                config["victim"]["muscleConfig"]["mslrConfig"]["freqBand"] = freqBands[args.target_model.lower()]
                
    if args.dataset is not None:
        config["target_dataset"]["name"] = args.dataset
        config["poison_dataset"]["name"] = args.dataset
        if args.dataset.lower() in ['agnews', 'miniagnews']:
            config["victim"]["num_classes"] = 4
        if args.dataset.lower() == 'lingspam' and config["attacker"]["poisoner"]["name"] in ["synbkd", "stylebkd"]:
            config["attacker"]["poisoner"]["longSent"] = True
            print("lingSpam should be set to Long Sentence")
    if args.poison_rate is not None:
        config["attacker"]["poisoner"]["poison_rate"] = args.poison_rate
    config = set_config(config)
    print(json.dumps(config, indent=4))
    
    set_seed(args.seed)
    
    
    main(config)
