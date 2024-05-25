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
import json
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
    parser.add_argument('--config_path', type=str, default='./llmConfigs/Badnets4GA+LoRA+MSLRConfig.json')
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
    print('tunable parameters:')
    for n, p in victim.llm.named_parameters():
        if p.requires_grad:
            print(n)
    print('victim model structure:')
    model_vis = Visualization(victim)
    model_vis.structure_graph()
    print('model state Dict')
    for n in victim.llm.state_dict().keys():
        print(n)
    
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

def deepspeedArgs(config):
    cmdArgs = argparse.Namespace()
    cmdArgs.deepspeed_config = './llmConfigs/deepspeedconf.json'
    config["attacker"]["train"]["cmdArgs"] = cmdArgs
    return config



if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    if args.target_model is not None:
        models = {
            'llama':"llama-2-7b-hf",
            'gpt':"gpt2-xl"
        }
        config["victim"]["path"] = models[args.target_model.lower()]
        config["victim"]["model"] = args.target_model.lower()
    if config["victim"]["model"] == "llama":
        freqBands = {
            'llama':[1, 2, 3, 4],
            'gpt':[1, 2, 3, 4],
        }
        if hasattr(config["victim"], "muscleConfig"):
            if hasattr(config["victim"]["muscleConfig"], "freqBand"):
                config["victim"]["muscleConfig"]["mslrConfig"]["freqBand"] = freqBands[args.target_model.lower()]
        
    if args.dataset is not None:
        config["target_dataset"]["name"] = args.dataset
        config["poison_dataset"]["name"] = args.dataset
        if args.dataset.lower() in ['agnews', 'miniagnews']:
            config["victim"]["num_classes"] = 4
        if args.dataset.lower() == 'lingspam' and config["attacker"]["poisoner"]["name"] in ["synbkd", "stylebkd"]:
            config["attacker"]["poisoner"]["longSent"] = True
            print("lingSpam belongs to Long Sentence")


    config = set_config(config)
    set_seed(args.seed)
    print(json.dumps(config, indent=4))
    if config["attacker"]["train"]["name"] == "amp":
        config = deepspeedArgs(config)
    main(config)
