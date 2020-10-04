import numpy as np
import tensorflow.keras as keras
import librosa
import math
import json
import os

# model.summary()

#################

test_folder_path = '../test'
json_path = '../test.json'

#################

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(test_folder_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5):
    
    # dictionary to store data
    data = {
        "mfcc": [],
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)  # rounding

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(test_folder_path)):

        # ensure that we are not at the root directory
        # if dirpath is not test_folder_path:
            
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path)

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s  # starting point for each segment
                    finish_sample = start_sample + num_samples_per_segment  # end point
                         
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                               sr=sr,
                                               n_fft=n_fft,
                                               n_mfcc=n_mfcc,
                                               hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        print("{}, segment:{}".format(file_path, s+1))

    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)



def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    # y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X


if __name__=="__main__":
    
    save_mfcc(test_folder_path, json_path, num_segments = 5)


    DATA_PATH = "../test.json"

    # load data
    X = load_data(DATA_PATH)
    print(X)

    # loading the trained model
    model = keras.models.load_model('my_model')

    # predicting the class as labels
    y_hat = model.predict(X)
    print(y_hat)

