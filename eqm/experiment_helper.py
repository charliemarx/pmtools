import sys
import time
import numpy as np
import warnings
import re
import datetime
from eqm.mip import ZeroOneLossMIP
from eqm.solution_pool import SolutionPool
from eqm.cplex_mip_helper import *
from pathlib import Path
from eqm.debug import ipsh


### printing

_LOG_TIME_FORMAT = "%m/%d/%y @ %I:%M %p"


def print_log(msg, print_flag = True):
    if print_flag:
        if isinstance(msg, str):
            print_str = '%s | %s' % (time.strftime(_LOG_TIME_FORMAT, time.localtime()), msg)
        else:
            print_str = '%s | %r' % (time.strftime(_LOG_TIME_FORMAT, time.localtime()), msg)
        print(print_str)
        sys.stdout.flush()


#### splitting dataset into  parts

TRIVIAL_PART_ID = "P01N01"
PART_PATTERN = "^P[0-9]{2}N[0-9]{2}$"
PART_ID_HANDLE = lambda p, n: 'P{:02}N{:02}'.format(p, n)
PART_PARSER = re.compile(PART_PATTERN)

PART_ID_PATTERN = "P[0-9]{2}N[0-9]{2}"
PART_ID_PARSER = re.compile("P[0-9]{2}N[0-9]{2}")

def filter_indices_to_part(u_idxs, part_id = TRIVIAL_PART_ID):
    """
    Given a list of indices, slices to a selection of the indices based on the part_id.
    :param u_idxs: The indices of unique points in a dataset.
    :param part_id: The part_id used to determine the slice. Specifies number of parts and the part to select.
    :return: The unique indices sliced to the specified part.
    """

    # extract info from params
    n_unique = len(u_idxs)
    part, n_parts = parse_part_id(part_id)

    # warn if splitting into more parts than can be made nonempty
    if n_parts > len(u_idxs):
        message = 'Splitting {} indices into {} parts. Some parts will be empty.'.format(len(u_idxs), n_parts)
        warnings.warn(message)

    # get indices to split on
    part_idxs = np.linspace(0, n_unique, n_parts + 1).astype(int)
    start = part_idxs[part - 1]
    end = part_idxs[part]

    return u_idxs[start:end]


def parse_part_id(part_id):
    """
    Given a part_id, extracts the number of parts and which part is selected
    :param part_id: The part_id to extract. Should be of the format specified at the top of this file.
    :return: The part selected and the number of parts.
    """
    # check that part_id is of the correct format
    assert bool(PART_PARSER.match(part_id)), 'invalid part_id.'

    # extract from the part_id
    str_nums = re.findall('\d+', part_id)
    part, n_parts = tuple(map(int, str_nums))

    # check that the information extracted is consistent
    assert part <= n_parts
    return part, n_parts


def get_part_id_helper(file_names):
    """
    :param file_names: list of file_names or a single file_name
    :return: dictionary containing keys for each partition, each partition containing a list of files matching the partition
    """

    if isinstance(file_names, (str, Path)):
        file_names = [file_names]

    # filter file names to files that exist on disk
    file_names = [Path(f) for f in file_names if f.exists()]
    file_names = list(set(file_names))

    # extra all part ids
    part_ids = [PART_ID_PARSER.findall(f.name) for f in file_names]
    part_ids = [p[0] for p in part_ids if len(p) > 0]

    # extract distinct counts
    part_counts = [re.findall('N[0-9]{2}', p) for p in part_ids]
    part_counts = [p[0][1:] for p in part_counts]
    distinct_part_counts = set([int(p) for p in part_counts])

    out = {}
    for n in distinct_part_counts:

        part_pattern = 'P[0-9]{2}N%02d' % n
        expected_parts = set(range(1, n + 1))

        matched_names = [re.search(part_pattern, f.name) for f in file_names]
        matched_names = [f for f in matched_names if f is not None]
        matched_parts = [f.group(0) for f in matched_names]
        matched_names = [f.string for f in matched_names]
        matched_files = [f for f in file_names if f.name in matched_names]
        modification_times = [datetime.datetime.fromtimestamp(f.stat().st_mtime) for f in matched_files]

        matched_parts = [re.search('P[0-9]{2,}', p) for p in matched_parts]
        matched_parts = [int(p.group(0)[1:]) for p in matched_parts]
        matched_parts = set(matched_parts)

        missing_parts = expected_parts.difference(matched_parts)
        missing_part_ids = ['P%02dN%02d' % (n, p) for p in missing_parts]

        out[n] = {
            'complete': len(missing_parts) == 0,
            'last_modification_time': max(modification_times) if len(modification_times) else None,
            'matched_parts': matched_parts,
            'matched_files': matched_files,
            'modification_times': modification_times,
            'missing_parts': missing_part_ids,
            }

    return out


##### analysis functions

def compute_error_rate_from_coefficients(W, X, Y):
    """
    todo: what does this do?
    :param W: A 2D-array of weights. Each row should be the weights of a single model.
    :param X: A 2D-array of features. Each row should be an instance.
    :param Y: A 1D-array of labels.
    :return:
    """
    assert len(Y) == len(X), "Number of labels inconsistent with number of features."
    assert W.shape[1] == X.shape[1], "Number of model parameters inconsistent with number of features."

    preds = np.sign(np.dot(W, X.T))
    return np.not_equal(preds, Y).mean(axis = 1)
