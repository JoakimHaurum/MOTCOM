
import os
import csv
import argparse


def init_config(config, default_config, name=None):
    """Initialise non-given config values with defaults"""
    if config is None:
        config = default_config
    else:
        for k in default_config.keys():
            if k not in config.keys():
                config[k] = default_config[k]
    if name and config['PRINT_CONFIG']:
        print('\n%s Config:' % name)
        for c in config.keys():
            print('%-20s : %-30s' % (c, config[c]))
    return config


def update_config(config):
    """
    Parse the arguments of a script and updates the config values for a given value if specified in the arguments.
    :param config: the config to update
    :return: the updated config
    """
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
            else:
                x = args[setting]
            config[setting] = x
    return config


def get_code_path():
    """Get base path where code is"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def validate_metrics_list(metrics_list):
    """Get names of metric class and ensures they are unique, further checks that the fields within each metric class
    do not have overlapping names.
    """
    metric_names = [metric.get_name() for metric in metrics_list]
    # check metric names are unique
    if len(metric_names) != len(set(metric_names)):
        raise MOTCOMEvalException('Code being run with multiple metrics of the same name')
    fields = []
    for m in metrics_list:
        fields += m.fields
    # check metric fields are unique
    if len(fields) != len(set(fields)):
        raise MOTCOMEvalException('Code being run with multiple metrics with fields of the same name')
    return metric_names


def write_detailed_results(details, metric_names, cls, output_folder):
    """Write detailed results to file"""
    sequences = details[0].keys()
    fields = ['seq'] + metric_names
    out_file = os.path.join(output_folder, cls + '_detailed.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for seq in sorted(sequences):
            if seq == 'COMBINED_SEQ':
                continue
            writer.writerow([seq] + sum([list(s[seq].values()) for s in details], []))


class MOTCOMEvalException(Exception):
    """Custom exception for catching expected errors."""
    ...
