import os
import librosa
import math
import json

DATASET_PATH = "../archive"
JSON_PATH = "./data.json"

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5):
    
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)  # rounding  
    
    # loop through all classes
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure that we are not at the root directory
        if dirpath is not dataset_path:
            
            # save semantic labels
            dirpath_components = dirpath.split("\\") # class/ambulance -> ["class","ambulance"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            
            # process files for a specific class
            for f in filenames:
                
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path)

                # signal = signal.T
                
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
                    
                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))
                        
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__=="__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=5)
    
                
                
    
    