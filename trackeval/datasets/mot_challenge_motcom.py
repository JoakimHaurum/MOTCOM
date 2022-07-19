import os
import csv
import configparser
import numpy as np
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import MOTCOMEvalException


class MOTChallengeMOTCOM(_BaseDataset):
    """Dataset class for MOT Challenge 2D bounding box MOTCOM calculation"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
            'OUTPUT_FOLDER': os.path.join(code_path, 'data/MOTCOM/mot_challenge/'),  # Where to save eval results 
            'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
            'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
            'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
            'PRINT_CONFIG': True,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/motcom_settings/OUTPUT_SUB_FOLDER
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
            'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/
                                      # If True, then the middle 'benchmark-split' folder is skipped.
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.benchmark = self.config['BENCHMARK']
        gt_set = self.config['BENCHMARK'] + '-' + self.config['SPLIT_TO_EVAL']
        self.gt_set = gt_set
        if not self.config['SKIP_SPLIT_FOL']:
            split_fol = gt_set
        else:
            split_fol = ''
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], split_fol)
        self.do_preproc = self.config['DO_PREPROC']

        self.output_fol = os.path.join(self.config['OUTPUT_FOLDER'], split_fol)
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # Get classes to eval
        self.valid_classes = ['pedestrian']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise MOTCOMEvalException('Attempted to evaluate an invalid class. Only pedestrian class is valid.')
        self.class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
                                       'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
                                       'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        self.seq_list, self.seq_lengths, self.seq_imagedim, self.seq_fps = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise MOTCOMEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        for seq in self.seq_list:
            curr_file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
            if not os.path.isfile(curr_file):
                print('GT file not found ' + curr_file)
                raise MOTCOMEvalException('GT file not found for sequence: ' + seq)

    def _get_seq_info(self):
        seq_list = []
        seq_lengths = {}
        seq_imagedim = {}
        seq_fps = {}
        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]

            # If sequence length is 'None' tries to read sequence length from .ini files.
            for seq, seq_length in seq_lengths.items():
                if seq_length is None:
                    ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                    if not os.path.isfile(ini_file):
                        raise MOTCOMEvalException('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
                    seq_imagedim[seq] = (int(ini_data['Sequence']['imWidth']), int(ini_data['Sequence']['imHeight']))
                    seq_fps[seq] = int(ini_data['Sequence']['frameRate'])
        else:
            if self.config["SEQMAP_FILE"]:
                seqmap_file = self.config["SEQMAP_FILE"]
            else:
                if self.config["SEQMAP_FOLDER"] is None:
                    seqmap_file = os.path.join(self.config['GT_FOLDER'], 'seqmaps', self.gt_set + '.txt')
                else:
                    seqmap_file = os.path.join(self.config["SEQMAP_FOLDER"], self.gt_set + '.txt')
            if not os.path.isfile(seqmap_file):
                print('no seqmap found: ' + seqmap_file)
                raise MOTCOMEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
            with open(seqmap_file) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0 or row[0] == '':
                        continue
                    seq = row[0]
                    if self.benchmark == "MOT17" and ("DPM" in seq or "FRCNN" in seq):
                        continue
                    seq_list.append(seq)
                    ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                    if not os.path.isfile(ini_file):
                        raise MOTCOMEvalException('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
                    seq_imagedim[seq] = (int(ini_data['Sequence']['imWidth']), int(ini_data['Sequence']['imHeight']))
                    seq_fps[seq] = int(ini_data['Sequence']['frameRate'])
        return seq_list, seq_lengths, seq_imagedim, seq_fps

    def _load_raw_file(self, seq, is_gt):
        """Load a gt file in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).
        """

        if not is_gt:
            raise MOTCOMEvalException("MOTCOM does not support using trackers")

        # File location
        file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file)

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets','gt_crowd_ignore_regions', 'gt_extras']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str( t+ 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            text = 'Ground-truth'
            raise MOTCOMEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))

        for t in range(num_timesteps):
            time_key = str(t+1)
            if time_key in read_data.keys():
                try:
                    time_data = np.asarray(read_data[time_key], dtype=np.float)
                except ValueError:
                    raise MOTCOMEvalException(
                        'Cannot convert gt data for sequence %s to float. Is data corrupted?' % seq)
                try:
                    raw_data['dets'][t] = np.atleast_2d(time_data[:, 2:6])
                    raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                except IndexError:
                    err = 'Cannot load gt data from sequence %s, because there is not enough ' \
                            'columns in the data.' % seq
                    raise MOTCOMEvalException(err)
                if time_data.shape[1] >= 8:
                    raw_data['classes'][t] = np.atleast_1d(time_data[:, 7]).astype(int)
                else:
                    raise MOTCOMEvalException(
                        'GT data is not in a valid format, there is not enough rows in seq %s, timestep %i.' % (
                            seq, t))
                gt_extras_dict = {'zero_marked': np.atleast_1d(time_data[:, 6].astype(int)), 'visibility': np.atleast_1d(time_data[:, 8].astype(float)), 'conf': np.atleast_1d(time_data[:, 6].astype(float))}
                raw_data['gt_extras'][t] = gt_extras_dict
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                gt_extras_dict = {'zero_marked': np.empty(0), 'visibility': np.empty(0), 'conf': np.empty(0)}
                raw_data['gt_extras'][t] = gt_extras_dict
                raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        key_map = {'ids': 'gt_ids',
                    'classes': 'gt_classes',
                    'dets': 'gt_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data


    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_gt_dets] : integers.
                    [gt_ids]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets]: list (for each timestep) of lists of detections.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Removes ground truth bounding boxes with non-positive width and height
                3) Removes bounding boxes fully outside the image, and crops bounding boxes partially outside the image.
                
            After the above preprocessing steps, this function also calculates the number of gt ids. 
                It also relabels gt to be contiguous and checks that ids are unique within each timestep.

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) There is no crowd ignore regions.
                3) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        distractor_class_names = ['person_on_vehicle', 'static_person', 'distractor', 'reflection']
        if self.benchmark == 'MOT20':
            distractor_class_names.append('non_mot_vehicle')
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'gt_dets', "visibility"]
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        num_gt_dets = 0
        
        imagedim = self.seq_imagedim[raw_data['seq']]
        for t in range(raw_data['num_timesteps']):

            # Get all data
            gt_ids = raw_data['gt_ids'][t]

            gt_dets = raw_data['gt_dets'][t]
            gt_dets[:,0] = gt_dets[:,0] - 1
            gt_dets[:,1] = gt_dets[:,1] - 1

            gt_classes = raw_data['gt_classes'][t]
            gt_zero_marked = raw_data['gt_extras'][t]['zero_marked']
            gt_visibility = raw_data['gt_extras'][t]['visibility']
            gt_conf = raw_data['gt_extras'][t]['conf']

            # Remove gt detections marked as to remove (zero marked), and also remove gt detections not in pedestrian
            # class (not applicable for MOT15)

            if self.do_preproc and self.benchmark != 'MOT15':
                gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
                                  (np.equal(gt_classes, cls_id)) & \
                                  (np.greater(gt_conf, 0)) & \
                                  (np.greater(gt_dets[:,2], 0)) & \
                                  (np.greater(gt_dets[:,3], 0))
            else:
                # There are no classes for MOT15
                gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
                                  (np.greater(gt_conf, 0)) & \
                                  (np.greater(gt_dets[:,2], 0)) & \
                                  (np.greater(gt_dets[:,3], 0))
                gt_visibility = np.ones_like(gt_visibility)
                

            
            # invalid checks - bbox fully outside image - add to gt_to_keep_mask
            outside_left = np.logical_and(np.less(gt_dets[:,0], 0), np.less(gt_dets[:,0] + gt_dets[:,2], 0))
            outside_top = np.logical_and(np.less(gt_dets[:,1], 0), np.less(gt_dets[:,1] + gt_dets[:,3], 0))
            outside_right = np.greater_equal(gt_dets[:,0], imagedim[0])
            outside_bottom = np.greater_equal(gt_dets[:,1], imagedim[1])

            # We negate the boolean vectors (~) to only keep those within the image
            gt_to_keep_mask = np.logical_and(gt_to_keep_mask, ~outside_left)
            gt_to_keep_mask = np.logical_and(gt_to_keep_mask, ~outside_top)
            gt_to_keep_mask = np.logical_and(gt_to_keep_mask, ~outside_right)
            gt_to_keep_mask = np.logical_and(gt_to_keep_mask, ~outside_bottom)
            
            # Determine whether bbox is partially outside image 
            zero_vector = np.zeros(len(gt_dets))

            ## Left side outside of image
            ### remove part outside of image. Using + since dets[0] is negative
            check_neg_left = np.less(gt_dets[:,0], 0)
            diff_width_left = gt_dets[:,2] + gt_dets[:,0]

            gt_dets[check_neg_left, 0] = zero_vector[check_neg_left]
            gt_dets[check_neg_left, 2] = diff_width_left[check_neg_left]


            ## Top side outside of image
            ### remove part outside of image. Using + since dets[1] is negative
            check_neg_top = np.less(gt_dets[:,1], 0)
            diff_height_top = gt_dets[:,3] + gt_dets[:,1]

            gt_dets[check_neg_top, 1] = zero_vector[check_neg_top]
            gt_dets[check_neg_top, 3] = diff_height_top[check_neg_top]

            
            ## Right side outside of image
            ### Remove part outside of image. New width is the distance from left corner to image border. -1 as it is 0 indexed
            check_neg_right = np.greater_equal(gt_dets[:,0]+gt_dets[:,2], imagedim[0])
            diff_width_right = imagedim[0] - gt_dets[:,0] - 1

            gt_dets[check_neg_right, 2] = diff_width_right[check_neg_right]
            

            ## Bottom side outside of image
            ### Remove part outside of image. New height is the distance from top to image border. -1 as it is 0 indexed
            check_neg_bottom = np.greater_equal(gt_dets[:,1]+gt_dets[:,3], imagedim[1])
            diff_height_bottom = imagedim[1] - gt_dets[:,1] - 1
             
            gt_dets[check_neg_bottom, 3] = diff_height_bottom[check_neg_bottom]

            
            # Verify that there are no bboxes with zero/negative width or height
            gt_to_keep_mask = np.logical_and(gt_to_keep_mask, np.greater(gt_dets[:,2], 0))
            gt_to_keep_mask = np.logical_and(gt_to_keep_mask, np.greater(gt_dets[:,3], 0))


            # Assign object properties
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['visibility'][t] = gt_visibility[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)

        # Record overview statistics.
        data['num_gt_dets'] = num_gt_dets
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        unique_ids = []
        for idx in range(len(data["gt_ids"])):
            unique_ids.extend(list(data["gt_ids"][idx]))
        unique_ids = list(set(unique_ids))      
        data["unique_ids"] = unique_ids
            
        id_frames = {str(gt_id): [] for gt_id in unique_ids}
        for f_idx in range(len(data["gt_ids"])):
            for gt_id in data["gt_ids"][f_idx]:
                id_frames[str(gt_id)].append(f_idx+1)  
        data["id_frames"] = id_frames

        id_visibility = {str(gt_id): [] for gt_id in unique_ids}
        id_dets = {str(gt_id): [] for gt_id in unique_ids}
        for f_idx in range(len(data["gt_ids"])):
            for idx, gt_id in enumerate(list(data["gt_ids"][f_idx])):
                val = data["visibility"][f_idx][idx]
                id_visibility[str(gt_id)].append(val)
                
                bbox = data["gt_dets"][f_idx][idx]
                id_dets[str(gt_id)].append(bbox)
        data["id_visibility"] = id_visibility
        data["id_dets"] = id_dets
        


        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)
        

        return data
