import os
import sys
import json
import random

import imageio
import torch
from torch.autograd import Variable
from torch import optim

import config
from anomaly_models import BetterAnomalyModel

def change_phase_factory():
    loss_history = []
    classification_history = []

    def change_phase(loss, classification):
        loss_history.append(loss)
        classification_history.append(classification)

        if loss.data.item() < 0.05:
            return True
        return False
    
    return change_phase

def is_anomaly(classification):
    if classification[0] > 0.5:
        return True
    return False

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    #1. Images parameters:
    images_folder = conf['generator']['images_folder']
    states_seq = conf['generator']['states_seq']
    max_nb_steps = conf['generator']['max_nb_steps']
    images_path_list = [os.path.join(images_folder, a) for a in os.listdir(images_folder) if a.endswith(".png")]
    images_list = [imageio.imread(im_path) for im_path in images_path_list]

    #2. Model parameters:
    seq_len = conf['model']['seq_len']
    gru_dropout = conf['model']['gru_dropout']
    output_dim = conf['model']['output_dim']
    model = BetterAnomalyModel(output_dim, gru_dropout, seq_len)
    
    #3. Optimizer parameters:
    lr = conf['optimizer']['lr']
    optimizer = optim.Adam(lr = lr, params = model.trainable_parameters())

    #4. Start the learning process
    nb_steps = 0
    states_seq_idx = -1
    change_phase = change_phase_factory()
    phase = "TRAINING"
    print("In TRAINING phase")

    while True:
        states_seq_idx += 1
        nb_steps += 1

        if phase == "TRAINING":
            next_frame = images_list[states_seq[states_seq_idx % len(states_seq)]]
            loss, classification = model.loss(next_frame, Variable(torch.Tensor([0, 1])))
            loss.backward()
            optimizer.step()

            if change_phase(loss, classification) or nb_steps > max_nb_steps:
                phase = "ANOMALY_DETECTION"
                print("In ANOMALY DETECTION phase")
            
        
        if phase == "ANOMALY DETECTION":
            if random.random() > 0.9:
                random_idx = random.choice(states_seq)
                if random_idx != states_seq_idx % len(states_seq):
                    print("ANOMALY INTRODUCED")
                next_frame = images_list[random_idx]
            else:    
                next_frame = images_list[states_seq[states_seq_idx % len(states_seq)]]
            classification = model.forward(next_frame)
            if is_anomaly(classification):
                print("ANOMALY DETECTED")

if __name__ == "__main__":
    main()