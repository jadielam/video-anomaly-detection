import torch
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms


from anomaly_models import BetterAnomalyModel
from anomaly_models import use_cuda
import config

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

def anomaly_detection_proc(frames_queue, conf):
    
    #0. Create transformation pipeline
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
    c_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    #1. Create the model
    seq_len = conf['model']['seq_len']
    gru_dropout = conf['model']['gru_dropout']
    model = BetterAnomalyModel(gru_dropout, seq_len)
    if use_cuda:
        model = model.cuda()
    
    change_phase = change_phase_factory()
    phase = "TRAINING"

    while True:
        message = frames_queue.get(True)
        if message == config.FINISH_SIGNAL:
            break
        _, frame = message

        if phase == "TRAINING":
            loss, classification = model.loss(frame, Variable(torch.Tensor([0, 1])))
            if change_phase(loss, classification):
                phase = "DETECTION"
                print("NOW IN DETECTION PHASE")
        elif phase == "DETECTION":
            classification = model.forward(frame)
            if is_anomaly(classification):
                print("ANOMALY DETECTED")