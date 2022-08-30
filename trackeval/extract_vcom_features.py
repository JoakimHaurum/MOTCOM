import time
import sys
import os
import argparse
import traceback
import pickle
import numpy as np
import cv2
import copy

from multiprocessing import freeze_support
from multiprocessing.pool import Pool
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import _timing
import utils
import trackeval


class VCOMFeaturesExtractor:
    @staticmethod
    def get_default_vcom_features_config():
        """Returns the default config values for VCOM features extraction"""
        code_path = utils.get_code_path()
        default_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.
            'IMAGES_FOLDER': os.path.join(code_path, "images"),
            'FEATURES_FOLDER': os.path.join(code_path, "VCOM_Features"),
            "BENCHMARK": "MOT17",
            "SEQUENCE": None,
            "EXTRACTOR": "RESNET18",
            "WEIGHTS_FOLDER": os.path.join(code_path, "weights"),
            "NUM_WORKERS_TORCH": 8,
            "BATCH_SIZE": 128,
            "BLUR_SIZE": 201,
            "BLUR_SIGMA": 38.0
            }
        return default_config

    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_vcom_features_config(), 'VCOMFeaturesExtractor')
        # Only run timing analysis if not run in parallel.
        if self.config['TIME_PROGRESS'] and not self.config['USE_PARALLEL']:
            _timing.DO_TIMING = True
            if self.config['DISPLAY_LESS_PROGRESS']:
                _timing.DISPLAY_LESS_PROGRESS = True

        self.tracker_name = "VCOMFeaturesExtractor"


    @_timing.time
    def extract(self, dataset_list):
        """Extract bounding boxes in the supplied datasets"""
        config = self.config
        dataset_names = [dataset.get_name() for dataset in dataset_list]
        output_res = {}
        output_msg = {}

        for dataset, dataset_name in zip(dataset_list, dataset_names):
            # Get dataset info about what to evaluate
            output_res[dataset_name] = {}
            output_msg[dataset_name] = {}
            seq_list, class_list = dataset.get_eval_info()
            if config["SEQUENCE"] is not None:
                seq_list = config["SEQUENCE"]
            print('\nExtracting features on %i sequence(s) on %s dataset' % (len(seq_list), dataset_name))

            try:
                time_start = time.time()
                if config['USE_PARALLEL']:
                    with Pool(config['NUM_PARALLEL_CORES']) as pool:
                        _eval_sequence = partial(extract_features_sequence, dataset=dataset, tracker = "", benchmark=config["BENCHMARK"], split=config["SPLIT_TO_EVAL"], img_folder = config["IMAGES_FOLDER"], features_folder=config["FEATURES_FOLDER"], weights_folder=config["WEIGHTS_FOLDER"],extractor_name=config["EXTRACTOR"], torch_workers=config["NUM_WORKERS_TORCH"], batch_size=config["BATCH_SIZE"], blur_width= config["BLUR_SIZE"], blur_sigma = config["BLUR_SIGMA"])
                        results = pool.map(_eval_sequence, seq_list)
                        res = dict(zip(seq_list, results))
                else:
                    res = {}
                    for curr_seq in sorted(seq_list):
                        res[curr_seq] = extract_features_sequence(curr_seq, dataset, "", class_list, config["BENCHMARK"], config["SPLIT_TO_EVAL"], config["IMAGES_FOLDER"], config["FEATURES_FOLDER"], config["WEIGHTS_FOLDER"],config["EXTRACTOR"], config["NUM_WORKERS_TORCH"], config["BATCH_SIZE"], config["BLUR_SIZE"], config["BLUR_SIGMA"])
                        

                # Print and output results in various formats
                if config['TIME_PROGRESS']:
                    print('\nAll sequences finished in %.2f seconds' % (time.time() - time_start))

                # Output for returning from function
                output_res[dataset_name][self.tracker_name] = res
                output_msg[dataset_name][self.tracker_name] = 'Success'

            except Exception as err:
                output_res[dataset_name][self.tracker_name] = None
                if type(err) == utils.MOTCOMEvalException:
                    output_msg[dataset_name][self.tracker_name] = str(err)
                else:
                    output_msg[dataset_name][self.tracker_name] = 'Unknown error occurred.'
                print('Was unable to extract bounding boxes')
                print(err)
                traceback.print_exc()
                if config['LOG_ON_ERROR'] is not None:
                    with open(config['LOG_ON_ERROR'], 'a') as f:
                        print(dataset_name, file=f)
                        print(traceback.format_exc(), file=f)
                        print('\n\n\n', file=f)
                if config['BREAK_ON_ERROR']:
                    raise err
                elif config['RETURN_ON_ERROR']:
                    return output_res, output_msg

        return output_res, output_msg


class VCOMDataset(Dataset):
    def __init__(self, img_dir, track_id, transform, data, blur_width, blur_sigma):
        self.img_dir = img_dir
        self.transform = transform

        self.blur_width = blur_width
        self.blur_sigma = blur_sigma
        self.to_pil = transforms.ToPILImage()
        self.blur = cv2.getGaussianKernel(blur_width, blur_sigma)
    
        self.frames = []
        self.bbs = []
        for idx in range(data["num_timesteps"]):
            if track_id in data["gt_ids"][idx]:
                t_id = list(data["gt_ids"][idx]).index(track_id)
                self.frames.append(idx+1)
                self.bbs.append(data["gt_dets"][idx][t_id])

        self.img_files = [str(i).zfill(6) + ".jpg" for i in self.frames]
        self.track_id = track_id
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_filepath = os.path.join(self.img_dir, self.img_files[idx])       
        image = default_loader(img_filepath)

        b_image = copy.copy(image)
        b_image = np.asarray(b_image)
        b_image = cv2.sepFilter2D(b_image, -1, self.blur, self.blur)

        b_image = self.transform(self.to_pil(b_image))
        t_image = self.transform(image)
        bb = self.bbs[idx]

        left = int(bb[0])
        top = int(bb[1])
        width = int(bb[2])
        height = int(bb[3])

        b_image[:,top:top+height,left:left+width] = t_image[:,top:top+height,left:left+width]
        
        return b_image, self.frames[idx]


TORCHVISION_MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
TORCHVISION_RESNET_URLS = models.resnet.model_urls

def get_extractor(extractor_name, weights_folder):

    if extractor_name.lower() in TORCHVISION_MODEL_NAMES:
        extractor  = models.__dict__[extractor_name.lower()](pretrained=False)
        model_url = TORCHVISION_RESNET_URLS[extractor_name.lower()]
    else:
        raise utils.MOTCOMEvalException('The requested feature extractor is not available in TorchVsion: ' + extractor_name)

    state_dict = torch.hub.load_state_dict_from_url(model_url, weights_folder)
    extractor.load_state_dict(state_dict)

    return extractor


@_timing.time
def extract_features_sequence(seq, dataset, tracker, class_list, benchmark, split, img_folder, features_folder, weights_folder, extractor_name, torch_workers, batch_size, blur_width, blur_sigma):
    """Function for evaluating a single sequence"""

    
    # Setup input and output paths
    seq_img_folder = os.path.join(img_folder, benchmark, split, seq, "img1")
    seq_features_folder = os.path.join(features_folder, benchmark, seq, "features")

    os.makedirs(seq_features_folder,exist_ok=True)
    
    # Setup feature extractor    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    extractor = get_extractor(extractor_name, weights_folder)
    extractor.fc = torch.nn.Identity()
    extractor.eval()
    extractor.to(device)
    
    # Setup data transformations
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    # load raw data 
    raw_data = dataset._load_raw_file(seq, is_gt=True)

    # Calculate features for every object in every frame (tracks)
    for cls in class_list:
        data = dataset.get_preprocessed_seq_data(raw_data, cls)

        unique_ids = []
        for idx in range(len(data["gt_ids"])):
            unique_ids.extend(list(data["gt_ids"][idx]))
        
        for track_id in data["unique_ids"]:
            dataloader = DataLoader(VCOMDataset(seq_img_folder, track_id, transform, data, blur_width, blur_sigma), batch_size = batch_size, num_workers=torch_workers)
            
            # The frame number is used as the dict key for the features
            data_features = dict()
            for batch in dataloader:
                t_images = batch[0]
                frames = np.asarray(batch[1])

                features = extractor(t_images.to(device)).detach().cpu().numpy()

                for idx, f in enumerate(frames):
                    data_features[f] = (features[idx],)

            # Store the features as a pickle file in the track img folder
            feat_path = os.path.join(seq_features_folder,'features_{}_{}.pickle'.format(extractor_name.lower(), track_id))
            with open(feat_path,'wb') as handle:
                pickle.dump(data_features,handle,protocol=pickle.HIGHEST_PROTOCOL)
            del data_features
            





if __name__ == '__main__':
    freeze_support()

    default_dataset_config = trackeval.datasets.MOTChallengeMOTCOM.get_default_dataset_config()
    default_features_config = VCOMFeaturesExtractor.get_default_vcom_features_config() 

    config = {**default_dataset_config, **default_features_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    features_config = {k: v for k, v in config.items() if k in default_features_config.keys()}
    features_config["SPLIT_TO_EVAL"] = dataset_config["SPLIT_TO_EVAL"]

    # Run code
    dataset_list = [trackeval.datasets.MOTChallengeMOTCOM(dataset_config)]
    features_extractor = VCOMFeaturesExtractor(features_config)

    features_extractor.extract(dataset_list)
