import scipy
import scipy.ndimage
import numpy as np
import os.path
import pickle

def take_samples(training_img, label_img):
    h = training_img.shape[0]
    w = training_img.shape[1]
    #positive samples
    positive_samples_pos = np.nonzero(label_img > 126)
    positive_samples_pos = np.array(positive_samples_pos)
    # 65x65 for filter
    #rows
    positive_samples_rows_to_keep1 = np.logical_not(positive_samples_pos[0] < 33)
    positive_samples_pos= positive_samples_pos[:,positive_samples_rows_to_keep1]
    positive_samples_rows_to_keep2 = np.logical_not(positive_samples_pos[0] > h - 33)
    positive_samples_pos= positive_samples_pos[:,positive_samples_rows_to_keep2]
    #cols
    positive_samples_cols_to_keep1 = np.logical_not(positive_samples_pos[1] < 33)
    positive_samples_pos= positive_samples_pos[:,positive_samples_cols_to_keep1]
    positive_samples_cols_to_keep2 = np.logical_not(positive_samples_pos[1] > w - 33)
    positive_samples_pos = positive_samples_pos[:,positive_samples_cols_to_keep2]
    positive_samples_size = positive_samples_pos.shape[1]
    #negative
    negative_samples_pos = np.nonzero(label_img < 100)
    negative_samples_pos = np.array(negative_samples_pos)
    # 65x65 for filter
    #rows
    negative_samples_rows_to_keep1 = np.logical_not(negative_samples_pos[0] < 33)
    negative_samples_pos= negative_samples_pos[:,negative_samples_rows_to_keep1]
    negative_samples_rows_to_keep2 = np.logical_not(negative_samples_pos[0] > h - 33)
    negative_samples_pos= negative_samples_pos[:,negative_samples_rows_to_keep2]
    #cols
    negative_samples_cols_to_keep1 = np.logical_not(negative_samples_pos[1] < 33)
    negative_samples_pos= negative_samples_pos[:,negative_samples_cols_to_keep1]
    negative_samples_cols_to_keep2 = np.logical_not(negative_samples_pos[1] > w - 33)
    negative_samples_pos = negative_samples_pos[:,negative_samples_cols_to_keep2]

    #sample negative samples
    valid_negative_sample_counts = negative_samples_pos.shape[1]
    chosen_negative_samples_pos_idx = np.random.choice(valid_negative_sample_counts, size=positive_samples_size,replace=False)
    chosen_negative_samples_pos = negative_samples_pos[:,chosen_negative_samples_pos_idx]
    negative_samples_size = chosen_negative_samples_pos.shape[1]

    assert(negative_samples_size == positive_samples_size)
    windows = []
    debug_counter_pos = 0
    #generate positive windows
    for row in positive_samples_pos.transpose():
        debug_counter_pos+=1
        window = training_img[row[0]-32:row[0]+33, row[1]-32:row[1]+33]
        windows.append(window.flatten())
    debug_counter = 0
    #generate negative windows
    for row in chosen_negative_samples_pos.transpose():
        debug_counter += 1
        window = training_img[row[0]-32:row[0]+33, row[1]-32:row[1]+33]
        windows.append(window.flatten())
    assert(debug_counter_pos == debug_counter)
    assert(debug_counter == positive_samples_size)
    #convert windows to features
    Xs = np.array(windows)
    #generate labels
    pos_labels = np.array([1, 0] * positive_samples_size).reshape(-1, 2)
    neg_labels = np.array([0 ,1] * negative_samples_size).reshape(-1, 2)
    Ys = np.concatenate((pos_labels, neg_labels))
    assert(Ys.shape[0] == Xs.shape[0])
    return (Xs, Ys)
def generate_training_data(training_manifest,labeling_manifest):
    with open(training_manifest, "rt") as f:
        training_images_files = f.read().splitlines()
    with open(labeling_manifest, "rt") as f:
        labeling_images_files = f.read().splitlines()
    manifest_prefix = os.path.dirname(training_manifest)
    labeling_prefix = os.path.dirname(labeling_manifest)
    for tr_img_file, lbl_img_file in zip(training_images_files, labeling_images_files):
        print("processing: ", tr_img_file, ", ", lbl_img_file)
        training_img = scipy.ndimage.imread(os.path.join(manifest_prefix, tr_img_file), mode="RGB")
        training_img = training_img[:,:,1]
        labeling_img = scipy.ndimage.imread(os.path.join(labeling_prefix, lbl_img_file), mode="L")
        Xs, Ys = take_samples(training_img, labeling_img)
        assert(Ys.shape[1] == 2)
        yield (Xs, Ys)

if __name__ == "__main__":
    # training_manifest = input("training manifest: ")
    # training_manifest = training_manifest.strip(" ")
    # labeling_manifest = input("labeling manifest: ")
    # labeling_manifest = labeling_manifest.strip(" ")
    training_manifest = "/Users/wujiaye/Documents/src/cse559/final_project/VesselSegmentation/DRIVE/training/images/manifest.txt"
    labeling_manifest = "/Users/wujiaye/Documents/src/cse559/final_project/VesselSegmentation/DRIVE/training/1st_manual/manifest.txt"
    with open("data_positive.pkl", "wb") as f:
        with open("data_negative.pkl", "wb") as f2:
            for Xs, Ys in generate_training_data(training_manifest, labeling_manifest):
                print(Xs[0:len(Xs)//2,:].shape[0], Xs[len(Xs)//2:,:].shape[0])
                print(Ys[0:len(Xs)//2,:].shape[0], Ys[len(Xs)//2:,:].shape[0])
                assert(Xs[0:len(Xs)//2,:].shape[0] == Xs[len(Xs)//2:,:].shape[0])
                pickle.dump((Xs[0:len(Xs)//2,:], Ys[0:len(Ys)//2,:]), f)
                pickle.dump((Xs[len(Xs)//2:,:], Ys[len(Ys)//2:,:]), f2)
