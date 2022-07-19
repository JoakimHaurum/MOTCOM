
import os
import pickle
import numpy as np
import scipy.spatial

from ._base_metric import _BaseMetric
from .. import _timing

class VCOM(_BaseMetric):

    def __init__(self, config=None):
        super().__init__()
        self.plottable = False
        self.array_labels = []
        self.float_array_fields = []
        self.integer_fields = []
        self.float_fields = ["VCOM"]
        self.fields = self.float_fields
        self.summary_fields = self.float_fields

        self.features_folder = config["FEATURES_FOLDER"] 
        self.extractor_name = config["EXTRACTOR"].lower()


    @_timing.time
    def eval_sequence(self, data):

        # Initialise results
        res = {}
        for field in self.float_fields:
            res[field] = 0

        total_frames, features, labels, frames = self.preload_VCOM_features(data)
           
        
        # Distance ratio range values
        lowlim = 0.01
        uplim = 1.0
        step_size = 0.01

        vcom_dists = []

        dists = [idx for idx in np.arange(lowlim, uplim+step_size, step_size)]
        for dist in dists:
            tmp_vcom = self.getVCOM(total_frames, features, labels, frames, distance_ratio=dist)
            vcom_dists.append(tmp_vcom)
        
        res["VCOM"] = np.mean(vcom_dists)
        return res 

    def getVCOM(self, total_frames, features, labels, frames, distance_ratio):
        
        features = features
        labels = labels
        frames = frames

        fdr_list = []
        lookup_frames = list(sorted(list(set(frames))))
        frame_counter = 0

        for f in total_frames:
            if not f in lookup_frames:
                continue
            center_frame = lookup_frames[frame_counter]
            next_frame = center_frame + 1
            if next_frame not in lookup_frames:
                continue
            
            frame_counter += 1

            cur_lf = [center_frame, next_frame]

            # Find objects present in the lookup frames
            center_indices, neighbour_indices, center_feats, neighbour_feats = self.lookupFeatures(cur_lf, center_frame, frames, features)
            
            if len(center_indices) == 0 or len(neighbour_indices) == 0:
                continue

            # Determine whether there is a full set of neighbours for each center ID
            _neighbour_labels = labels[neighbour_indices]
            _center_labels = labels[center_indices]

            results_dict = {key: -1 for key in _center_labels}
            
            dists = scipy.spatial.distance.cdist(center_feats, neighbour_feats, "euclidean")

            for cur_idx, cur_label in enumerate(_center_labels):

                if cur_label not in _neighbour_labels:
                    # If the object in focus is not present in the neighbour frame, remove it from the calculation
                    results_dict.pop(cur_label, None)                
                else:
                    # Sort based on the closest distance
                    sorted_dists_idx = np.argsort(dists[cur_idx])
                    sorted_dists = dists[cur_idx][sorted_dists_idx]

                    # Check instance within a distance_ratio ball
                    dist_ratio = sorted_dists[0] + sorted_dists[0] * distance_ratio

                    valid_dists = sorted_dists <= dist_ratio
                    valid_idx = sorted_dists_idx[valid_dists]

                    # Determine number of true/total assignments
                    total_idx_count = len(valid_idx)
                    true_count = sum([1 if _neighbour_labels[idx] == cur_label else 0 for idx in valid_idx])
                    results_dict[cur_label] = (float(true_count), float(total_idx_count))

            if len(results_dict) == 0:
                continue

            frame_fdr = self.mean_fdr_frame(results_dict)
            fdr_list.append(frame_fdr)

        return np.mean(fdr_list)

    def mean_fdr_frame(self, results):
        """
        Calculates the average False Detection Rate for the frame

        Arguments:
            results: dict of IDs with number of True Positives and Total Positives

        Return:
            FDR: The False Detection Rate for the frame
        """
        out = {label:[] for label in results}
        for key in results:
            if isinstance(results[key], (list, tuple)):
                acc = 1 - (results[key][0] / results[key][1])
            else:
                acc = 1
            out[key].append(acc)

        FDR = np.mean([out[acc] for acc in out])
        return FDR



    def preload_VCOM_features(self, data):
                
        total_frames = [idx for idx in range(1, data["num_timesteps"]+1)]
        tracks = self.loadPickledFeatures(data["benchmark"], data["seq"])

        ids = list(tracks)
        indices = [(i,x) for i in ids for x in tracks[i] if not isinstance(tracks[i][x][0],np.ndarray)]
        for i in indices:
            del tracks[i[0]][i[1]]

        features = np.array([tracks[i][x][0] for i in ids for x in tracks[i]])
        labels = np.array([i for i in ids for x in tracks[i]])
        frames = np.hstack([list(tracks[i]) for i in ids]).astype(int) # The track dict keys are the frames

        return total_frames, features, labels, frames


    def lookupFeatures(self, cur_lf, center_frame, frames, features):
        center_indices = np.hstack([np.where(frames==lf)[0] for lf in cur_lf if lf == center_frame])
        neighbour_indices = np.hstack([np.where(frames==lf)[0] for lf in cur_lf if lf != center_frame])
        center_feats = features[center_indices]
        neighbour_feats = features[neighbour_indices]

        return center_indices, neighbour_indices, center_feats, neighbour_feats
                

    def loadPickledFeatures(self, benchmark, seq):
        """
        Reads the features from the features.pickle file in every object sub-directory

        Returns:
            dict of dicts: with object id as outer key, frame as innter key and feature vectors as values
        """
        all_features = dict()
        full_features_folder = os.path.join(self.features_folder, benchmark, seq, "features")
        for filename in os.listdir(full_features_folder):
            if "features_{}".format(self.extractor_name) in filename:
                target_id = int(filename.split(".pickle")[0].split("_")[-1])
                with open(os.path.join(full_features_folder,filename),'rb') as handle:
                    all_features[target_id] = pickle.load(handle)   
        return all_features
