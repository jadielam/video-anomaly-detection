import random
import numpy as np
import cv2
import xgboost as xgb

import config

def is_anomaly(classification):
    return classification[0] == 1

def find_geometric_figure_data(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                param1 = 100, param2 = 30, minRadius = 1, maxRadius = 5000)
    if circles is not None:
        circles = circles[0,:]
        for (x, y, r) in circles:
            return x, y, r, -1
    return -1, -1, -1, -1

def extract_features(last_k_xs, last_k_ys, last_k_rs, last_k_os):
    return last_k_xs + last_k_ys + last_k_rs + last_k_os

class FeatureExtractor():
    def __init__(self, k):
        self._k = k
        self._last_k_xs = []
        self._last_k_ys = []
        self._last_k_rs = []
        self._last_k_os = []
        self._min_x = 99999
        self._max_x = -1
        self._min_y = 99999
        self._max_y = -1
        self._min_r = 99999
        self._max_r = -1
        self._min_o = 99999
        self._max_o = -1
    
    def step(self, frame):
        '''
        Returns the features extracted for that frame
        '''
        # Finding coordinates of geometric figure
        x, y, r, o = find_geometric_figure_data(frame)
        
        # Updating maximums and minimums
        for entry in zip([x, y, r, o], [self._min_x, self._min_y, self._min_r, self._min_o]):
            if entry[0] < entry[1]:
                entry[1] = entry[0]
        for entry in zip([x, y, r, o], [self._max_x, self._max_y, self._max_r, self._max_o]):
            if entry[0] > entry[1]:
                entry[1] = entry[0]
        
        # Updating rolling windows
        self._last_k_xs = self._last_k_xs[1:self._k] + [x]
        self._last_k_ys = self._last_k_ys[1:self._k] + [y]
        self._last_k_rs = self._last_k_rs[1:self._k] + [r]
        self._last_k_os = self._last_k_os[1:self._k] + [o]

        # Extracting features
        good_features = extract_features(
            self._last_k_xs, 
            self._last_k_ys, 
            self._last_k_rs, 
            self._last_k_os)
        
        bad_features = extract_features(
            self._last_k_xs, 
            self._last_k_ys, 
            self._last_k_rs, 
            self._last_k_os
        )
        return good_features, bad_features

def train_model(X_train, y_train, X_val, y_val,             
                parameters = {},
                num_boost_round = 10000, 
                early_stopping_rounds = 50,
                num_class = 2):
    
    default_params = {
                    "objective": "multi:softmax",
                    "num_class": num_class,
                    "booster": "gbtree",
                    "eval_metric": "mlogloss",
                    "eta": 0.02,
                    "max_depth": 4,
                    "subsample": 0.6,
                    "colsample_bytree": 0.6,
                    "min_child_weights": 1,
                    "silent": 1,
                    "gamma": 0,
                    "seed": 42
    }

    model_params = default_params.copy()
    model_params.update(parameters)
    X_y_train = xgb.DMatrix(X_train, y_train, missing = np.NaN)
    X_y_val = xgb.DMatrix(X_val, y_val, missing = np.NaN)
    watchlist = [(X_y_train, 'train'), (X_y_val, 'eval')]
    progress = dict()

    xgb_model = xgb.train(model_params, X_y_train, num_boost_round, 
                        evals = watchlist, 
                        early_stopping_rounds = early_stopping_rounds,
                        evals_result = progress)

    parameters['n_iterations'] = num_boost_round
    return xgb_model, parameters, progress

def predict_with_model(xgb_model, X_to_predict):
    X_predict = xgb.DMatrix(X_to_predict, missing = np.NaN)
    predictions = xgb_model.predict(X_predict)
    return predictions

def anomaly_detection_proc(frames_queue, conf):
    
    k = 30
    nb_steps = 0
    states_seq_idx = -1
    phase = 'TRAINING'

    print("In {} phase".format(phase))
    feature_extractor = FeatureExtractor(k)
    X_train_l = []
    Y_train_l = []

    while True:
        states_seq_idx += 1
        nb_steps += 1

        if phase == 'TRAINING':
            random_idx = ran
            good_next_frame = 
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