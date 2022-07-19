import csv
import os
import traceback
import numpy as np
from abc import ABC, abstractmethod
from .. import _timing
from ..utils import MOTCOMEvalException


class _BaseDataset(ABC):
    @abstractmethod
    def __init__(self):
        self.seq_list = None
        self.class_list = None
        self.output_fol = None
        self.output_sub_fol = None

    # Functions to implement:

    @staticmethod
    @abstractmethod
    def get_default_dataset_config():
        ...

    @abstractmethod
    def _load_raw_file(self, seq, is_gt):
        ...

    @_timing.time
    @abstractmethod
    def get_preprocessed_seq_data(self, raw_data, cls):
        ...

    # Helper functions for all datasets:

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def get_name(self):
        return self.get_class_name()

    def get_output_fol(self, motcom_settings):
        return os.path.join(self.output_fol, motcom_settings, self.output_sub_fol)

    def get_eval_info(self):
        """Return info about the dataset needed for the Evaluator"""
        return self.seq_list, self.class_list

    @staticmethod
    def _load_simple_text_file(file, time_col=0, id_col=None, remove_negative_ids=False, valid_filter=None,
                               crowd_ignore_filter=None, convert_filter=None, force_delimiters=None):
        """ Function that loads data which is in a commonly used text file format.
        Assumes each det is given by one row of a text file.
        There is no limit to the number or meaning of each column,
        however one column needs to give the timestep of each det (time_col) which is default col 0.

        The file dialect (deliminator, num cols, etc) is determined automatically.
        This function automatically separates dets by timestep,
        and is much faster than alternatives such as np.loadtext or pandas.

        If remove_negative_ids is True and id_col is not None, dets with negative values in id_col are excluded.
        These are not excluded from ignore data.

        valid_filter can be used to only include certain classes.
        It is a dict with ints as keys, and lists as values,
        such that a row is included if "row[key].lower() is in value" for all key/value pairs in the dict.
        If None, all classes are included.

        crowd_ignore_filter can be used to read crowd_ignore regions separately. It has the same format as valid filter.

        convert_filter can be used to convert value read to another format.
        This is used most commonly to convert classes given as string to a class id.
        This is a dict such that the key is the column to convert, and the value is another dict giving the mapping.

        Returns read_data and ignore_data.
        Each is a dict (with keys as timesteps as strings) of lists (over dets) of lists (over column values).
        Note that all data is returned as strings, and must be converted to float/int later if needed.
        Note that timesteps will not be present in the returned dict keys if there are no dets for them
        """

        if remove_negative_ids and id_col is None:
            raise MOTCOMEvalException('remove_negative_ids is True, but id_col is not given.')
        if crowd_ignore_filter is None:
            crowd_ignore_filter = {}
        if convert_filter is None:
            convert_filter = {}
        try:
            fp = open(file)
            read_data = {}
            crowd_ignore_data = {}
            fp.seek(0, os.SEEK_END)
            # check if file is empty
            if fp.tell():
                fp.seek(0)
                dialect = csv.Sniffer().sniff(fp.readline(), delimiters=force_delimiters)  # Auto determine structure.
                dialect.skipinitialspace = True  # Deal with extra spaces between columns
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    try:
                        # Deal with extra trailing spaces at the end of rows
                        if row[-1] in '':
                            row = row[:-1]
                        timestep = str(int(float(row[time_col])))
                        # Read ignore regions separately.
                        is_ignored = False
                        for ignore_key, ignore_value in crowd_ignore_filter.items():
                            if row[ignore_key].lower() in ignore_value:
                                # Convert values in one column (e.g. string to id)
                                for convert_key, convert_value in convert_filter.items():
                                    row[convert_key] = convert_value[row[convert_key].lower()]
                                # Save data separated by timestep.
                                if timestep in crowd_ignore_data.keys():
                                    crowd_ignore_data[timestep].append(row)
                                else:
                                    crowd_ignore_data[timestep] = [row]
                                is_ignored = True
                        if is_ignored:  # if det is an ignore region, it cannot be a normal det.
                            continue
                        # Exclude some dets if not valid.
                        if valid_filter is not None:
                            for key, value in valid_filter.items():
                                if row[key].lower() not in value:
                                    continue
                        if remove_negative_ids:
                            if int(float(row[id_col])) < 0:
                                continue
                        # Convert values in one column (e.g. string to id)
                        for convert_key, convert_value in convert_filter.items():
                            row[convert_key] = convert_value[row[convert_key].lower()]
                        # Save data separated by timestep.
                        if timestep in read_data.keys():
                            read_data[timestep].append(row)
                        else:
                            read_data[timestep] = [row]
                    except Exception:
                        exc_str_init = 'In file %s the following line cannot be read correctly: \n' % os.path.basename(
                            file)
                        exc_str = ' '.join([exc_str_init]+row)
                        raise MOTCOMEvalException(exc_str)
            fp.close()
        except Exception:
            print('Error loading file: %s, printing traceback.' % file)
            traceback.print_exc()
            raise MOTCOMEvalException(
                'File %s cannot be read because it is either not present or invalidly formatted' % os.path.basename(
                    file))
        return read_data, crowd_ignore_data

    @staticmethod
    def _check_unique_ids(data, after_preproc=False):
        """Check the requirement that the gt_ids are unique per timestep"""
        gt_ids = data['gt_ids']
        for t, gt_ids_t in enumerate(gt_ids):
            if len(gt_ids_t) > 0:
                unique_ids, counts = np.unique(gt_ids_t, return_counts=True)
                if np.max(counts) != 1:
                    duplicate_ids = unique_ids[counts > 1]
                    exc_str_init = 'Ground-truth has the same ID more than once in a single timestep ' \
                                   '(seq: %s, frame: %i, ids:' % (data['seq'], t+1)
                    exc_str = ' '.join([exc_str_init] + [str(d) for d in duplicate_ids]) + ')'
                    if after_preproc:
                        exc_str_init += '\n Note that this error occurred after preprocessing (but not before), ' \
                                        'so ids may not be as in file, and something seems wrong with preproc.'
                    raise MOTCOMEvalException(exc_str)

