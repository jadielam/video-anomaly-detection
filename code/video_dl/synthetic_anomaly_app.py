import os
import sys
import json
import random

from PIL import Image
import torch
from torch.autograd import Variable
from torch import optim
import torchvision.transforms as transforms

import config
from anomaly_models import BetterAnomalyModel
from anomaly_models import use_cuda

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
    images_path_list.sort()
    images_list = [Image.open(im_path) for im_path in images_path_list]
    
    #1.1 Transform image list:
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
    c_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    images_list = [c_transforms(a) for a in images_list]
    images_list = [a[:3,:,:].cuda() for a in images_list]

    #2. Model parameters:
    seq_len = conf['model']['seq_len']
    gru_dropout = conf['model']['gru_dropout']
    model = BetterAnomalyModel(gru_dropout, seq_len)
    if use_cuda:
        model = model.cuda()
    
    #3. Optimizer parameters:
    l_r = conf['optimizer']['lr']
    optimizer = optim.Adam(lr = l_r, params = model.trainable_parameters())

    #4. Start the learning process
    nb_steps = 0
    states_seq_idx = -1
    change_phase = change_phase_factory()
    phase = "TRAINING"
    print("In TRAINING phase")

    if use_cuda:
        seq_pack = torch.zeros((seq_len, images_list[0].shape[0], images_list[0].shape[1], images_list[0].shape[2])).cuda()
    else:
        seq_pack = torch.zeros((seq_len, images_list[0].shape[0], images_list[0].shape[1], images_list[0].shape[2]))

    while True:
        states_seq_idx += 1
        nb_steps += 1

        if phase == "TRAINING":
            next_frame = images_list[states_seq[states_seq_idx % len(states_seq)]].unsqueeze(0)
            frame_sequence = torch.cat([next_frame, seq_pack[:-1]])
            #frame_sequence = torch.cat([seq_pack[1:], next_frame])
            seq_pack = frame_sequence.clone()
            frame_sequence = Variable(frame_sequence)
            loss, classification = model.loss(frame_sequence, Variable(torch.Tensor([0, 1])).expand(seq_len, 2))
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
            
            frame_sequence = torch.cat([next_frame, seq_pack[:-1]])
            #frame_sequence = torch.cat([seq_pack[1:], next_frame])
            seq_pack = frame_sequence.clone()
            frame_sequence = Variable(frame_sequence)
            
            classification = model.forward(frame_sequence)
            if is_anomaly(classification):
                print("ANOMALY DETECTED")

if __name__ == "__main__":
    main()