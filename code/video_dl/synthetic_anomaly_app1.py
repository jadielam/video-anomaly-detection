import os
import sys
import json
import random
import numpy as np

import cv2
import config


def find_geometric_figure_data(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
        circles = circles[0,:]
        for (x, y, r) in circles:
            return x, y, r, -1

def extract_features(last_k_xs, last_k_ys, last_k_rs, last_k_os):
    return last_k_xs + last_k_ys + last_k_rs + last_k_os

class FeatureExtractor():
    def __init__(self, k):
        self._k = k
        self._last_k_xs = []
        self._last_k_ys = []
        self._last_k_rs = []
        self._last_k_os = []
    
    def step(frame):
        '''
        Returns the features extracted for that frame
        '''
        x, y, r, o = find_geometric_figure_data(frame)

        self._last_k_xs = last_k_xs[1:self._k] + [x]
        self._last_k_ys = last_k_ys[1:self._k] + [y]
        self._last_k_rs = last_k_rs[1:self._k] + [r]
        self._last_k_os = last_k_os[1:self._k] + [o]

        return extract_features(self._last_k_xs, self._last_k_ys, 
                                self._last_k_rs, self._last_k_os)

    
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

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    #1. Parameters initialization
    k = 30
    nb_steps = 0
    states_seq_idx = -1
    phase = "TRAINING"
    images_folder = conf['generator']['images_folder']
    states_seq = conf['generator']['states_seq']
    max_nb_steps = conf['generator']['max_nb_steps']

    #2. Read images
    images_path_list = [os.path.join(images_folder, a) for a in os.listdir(images_folder) if a.endswith(".jpg")]
    images_path_list.sort()
    images_list = [cv2.imread(im_path) for im_path in images_path_list]

    #3. Learning process
    print("In TRAINING phase")
    feature_extractor = FeatureExtractor()
    X_train_l = []
    Y_train_l = []

    while True:
        states_seq_idx += 1
        nb_steps += 1
        target = 0
        if phase == "TRAINING":
            if random.random() > 0.5:
                random_idx = random.choice(range(len(images_list)))
                if random_idx != states_seq[states_seq_idx % len(states_seq)]:
                    target = 1
                next_frame = images_list[random_idx]
            else:
                next_frame = images_list[states_seq[states_seq_idx % len(states_seq)]]
            
            if nb_steps > k:
                features = feature_extractor.step(next_frame)
                X_train_l.append(features)
                Y_train_l.append(target)

            if nb_steps > max_nb_steps:
                #1. Build model
                model, _, _ = train_model(
                    np.array(X_train_l[200:]), 
                    np.array(Y_train_l[200:]),
                    np.array(X_train_l[:200]),
                    np.array(Y_train_l[:200])
                )
                
                #2. Announce change to anomaly detection phase
                phase = "ANOMALY_DETECTION"
                print("In ANOMALY DETECTION phase")
        
        if phase == "ANOMALY_DETECTION":
            if random.random() > 0.9:
                random_idx = random.choice(range(len(images_list)))
                if random_idx != states_seq[states_seq_idx % len(states_seq)]:
                    print("ANOMALY INTRODUCED")
                next_frame = images_list[random_idx]
            else:    
                next_frame = images_list[states_seq[states_seq_idx % len(states_seq)]]
            
            features = feature_extractor.step(next_frame)
            
            classification = predict_with_model(model, features)
            print(classification)
            #if is_anomaly(classification):
            #    print("ANOMALY DETECTED")

if __name__ == "__main__":
    main()