import time
import traceback
from multiprocessing.pool import Pool
from functools import partial
import os
from . import utils
from .utils import MOTCOMEvalException
from . import _timing


class EvaluatorMOTCOM:
    """Evaluator class for evaluating different metrics for different datasets"""

    @staticmethod
    def get_default_eval_config():
        """Returns the default config values for evaluation"""
        code_path = utils.get_code_path()
        default_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.

            'PRINT_RESULTS': True,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,

            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': True,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_eval_config(), 'Eval')
        # Only run timing analysis if not run in parallel.
        if self.config['TIME_PROGRESS'] and not self.config['USE_PARALLEL']:
            _timing.DO_TIMING = True
            if self.config['DISPLAY_LESS_PROGRESS']:
                _timing.DISPLAY_LESS_PROGRESS = True

        self.motcom_settings = "bbox{}_ycoord{}_{}".format(int(config["BBOX_BASED_OCC"]), int(config["Y_COORD_OCC"]), config["EXTRACTOR"].lower())

    @_timing.time
    def evaluate(self, dataset_list, metrics_list):
        """Evaluate a set of metrics on a set of datasets"""
        config = self.config
        metrics_list = metrics_list
        metric_names = utils.validate_metrics_list(metrics_list)
        dataset_names = [dataset.get_name() for dataset in dataset_list]
        output_res = {}
        output_msg = {}

        for dataset, dataset_name in zip(dataset_list, dataset_names):
            # Get dataset info about what to evaluate
            output_res[dataset_name] = {}
            output_msg[dataset_name] = {}
            seq_list, class_list = dataset.get_eval_info()
            print('\nEvaluating MOTCOM on %i sequence(s) for %i class(es) on %s dataset using the following '
                  'metrics: %s\n' % (len(seq_list), len(class_list), dataset_name,
                                     ', '.join(metric_names)))

            try:
                # Evaluate each sequence in parallel or in series.
                # returns a nested dict (res), indexed like: res[seq][class][metric_name][sub_metric field]
                # e.g. res[seq_0001][pedestrian][OCOM]
                
                output_fol = dataset.get_output_fol(self.motcom_settings)
                time_start = time.time()
                if config['USE_PARALLEL']:
                    with Pool(config['NUM_PARALLEL_CORES']) as pool:
                        _eval_sequence = partial(eval_sequence, dataset=dataset, benchmark=dataset.benchmark,
                                                    class_list=class_list, metrics_list=metrics_list,
                                                    metric_names=metric_names)
                        results = pool.map(_eval_sequence, seq_list)
                        res = dict(zip(seq_list, results))
                else:
                    res = {}
                    for curr_seq in sorted(seq_list):
                        res[curr_seq] = eval_sequence(curr_seq, dataset, dataset.benchmark, class_list, metrics_list,
                                                        metric_names)

                # Print and output results in various formats
                if config['TIME_PROGRESS']:
                    print('\nAll sequences finished in %.2f seconds' % (time.time() - time_start))
                for c_cls in class_list:  # class_list + combined classes if calculated
                    details = []
                    if config['OUTPUT_EMPTY_CLASSES']:
                        for metric, metric_name in zip(metrics_list, metric_names):
                            table_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value
                                                in res.items()}

                            if config['PRINT_RESULTS']:
                                metric.print_table(table_res, self.motcom_settings, c_cls)
                            if config['OUTPUT_DETAILED']:
                                details.append(metric.detailed_results(table_res))
                        if config['OUTPUT_DETAILED']:
                            utils.write_detailed_results(details, metric_names, c_cls, output_fol)

                # Output for returning from function
                output_res[dataset_name][self.motcom_settings] = res
                output_msg[dataset_name][self.motcom_settings] = 'Success'

            except Exception as err:
                output_res[dataset_name][self.motcom_settings] = None
                if type(err) == MOTCOMEvalException:
                    output_msg[dataset_name][self.motcom_settings] = str(err)
                else:
                    output_msg[dataset_name][self.motcom_settings] = 'Unknown error occurred.'
                print('Was unable to compute MOTCOM')
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


@_timing.time
def eval_sequence(seq, dataset, benchmark, class_list, metrics_list, metric_names):
    """Function for evaluating a single sequence"""

    raw_data = dataset._load_raw_file(seq, is_gt=True)
    seq_res = {}
    for cls in class_list:
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls)
        data["benchmark"] = benchmark
        data["image_size"] = dataset.seq_imagedim[seq]
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[cls][met_name] = metric.eval_sequence(data)
    return seq_res
