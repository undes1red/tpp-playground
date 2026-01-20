import numpy as np

from src.toolbox.metrics.otd import find_alignment_mc


def edit_distance_mt_mc(ref, decoded, del_cost, trans_cost, n_types):
    """

    :param list ref:
    :param list decoded:
    :param np.ndarray del_cost:
    :param float trans_cost:
    :param int n_types:
    """
    num_cost = len(del_cost)

    distances = np.zeros(shape=[num_cost], dtype=np.float32)
    total_trans_cost = np.zeros(shape=[num_cost], dtype=np.float32)
    num_true = np.zeros(shape=[num_cost], dtype=np.int32)
    num_del = np.zeros(shape=[num_cost], dtype=np.int32)
    num_ins = np.zeros(shape=[num_cost], dtype=np.int32)
    num_align = np.zeros(shape=[num_cost], dtype=np.int32)

    seq_per_types = [[list(), list()] for _ in range(n_types)]
    for seq_idx, seq in enumerate([ref, decoded]):
        for token in seq:
            event_type = token['type_event']
            if event_type >= n_types:
                continue
            seq_per_types[event_type][seq_idx].append(token['time_since_start'])

    for type_idx in range(n_types):
        ref_time = np.array(seq_per_types[type_idx][0])
        decoded_time = np.array(seq_per_types[type_idx][1])
        align_pairs, min_distance = find_alignment_mc(
            ref_time, decoded_time, del_cost, trans_cost)
        for cost_idx in range(num_cost):
            align_pairs_per_cost = align_pairs[cost_idx]
            min_distance_per_cost = min_distance[cost_idx]
            num_align[cost_idx] += len(align_pairs_per_cost)
            num_true[cost_idx] += len(ref_time)
            n_ins_per_cost = len(decoded_time) - len(align_pairs_per_cost)
            n_del_per_cost = len(ref_time) - len(align_pairs_per_cost)
            num_ins[cost_idx] += n_ins_per_cost
            num_del[cost_idx] += n_del_per_cost
            distances[cost_idx] += min_distance_per_cost
            total_trans_cost[cost_idx] += min_distance_per_cost\
                                          - del_cost[cost_idx]*(n_ins_per_cost+n_del_per_cost)

    return distances, total_trans_cost, num_true, num_del, num_ins, num_align


def max_triangle_1d(width, time_stamps, heights):
    """

    :param float width:
    :param np.ndarray time_stamps:
    :param np.ndarray heights:
    """
    distances = np.outer(time_stamps, np.ones(shape=[len(time_stamps)], dtype=np.float32))
    distances = np.abs(distances - time_stamps)
    effective_distances = width - distances
    effective_distances[effective_distances < 0] = 0.0
    gain = heights / width
    values = (effective_distances * gain).sum(axis=1)

    highest_idx = np.argmax(values)
    highest_value = values[highest_idx]

    return int(highest_idx), highest_value


def max_triangle_2d(width, time_stamps, heights, lens):
    """

    :param float width:
    :param np.ndarray time_stamps: shape=[n, m], dtype=np.float32
    :param np.ndarray heights: shape=[n], dtype=np.float32
    :param np.ndarray lens: shape=[n], dtype=np.int32
    """
    n, m = time_stamps.shape
    flatten_time_stamps = time_stamps.reshape(n * m)
    distances = np.outer(flatten_time_stamps, np.ones(shape=[n*m], dtype=np.float32))
    distances = np.abs(distances - flatten_time_stamps)
    distances = width - distances
    distances[distances < 0] = 0.0

    # shape=[n1, m1, n2, m2]
    distances = distances.reshape(n, m, n, m)
    # shape=[n1, m1, m2, n2]
    distances = distances.transpose([0, 1, 3, 2])
    gain = heights / width
    distances = distances * gain
    # shape=[n1, m1, n2, m2]
    distances = distances.transpose([0, 1, 3, 2])
    # shape=[m, n]
    len_mask = np.outer(np.arange(m, dtype=np.int32), np.ones(shape=[n], dtype=np.int32))
    len_mask = len_mask < lens
    # shape=[n, m]
    len_mask = len_mask.transpose([1, 0])

    distances[:, :, ~len_mask] = 0.0
    # shape=[n1, m1, n2]
    distances_each = np.max(distances, axis=3)
    # shape=[n1, m1]
    distances_sum = distances_each.sum(axis=2)
    distances_sum[~len_mask] = 0.0
    # shape=[n1 * m1]
    distances_sum = distances_sum.reshape(n * m)

    max_idx = np.argmax(distances_sum)
    max_value = distances_sum[max_idx]
    max_idx1 = max_idx // m
    max_idx2 = max_idx % m

    # shape=[n2, m2]
    distances_sub_mat = distances[max_idx1, max_idx2]
    choices = distances_sub_mat.argmax(axis=1)
    choice_mask = distances_sub_mat.max(axis=1) <= 0.0
    choices[choice_mask] = -1

    return [max_idx1, max_idx2], max_value, choices


def concat_pad_mat(a, pad=0):
    """

    :param list[np.ndarray] a:
    :param float pad:
    :rtype: np.ndarray
    """
    max_len = max([len(item) for item in a])
    n = len(a)
    rst = np.full(shape=[n, max_len], fill_value=pad, dtype=a[0].dtype)
    for row_idx, row in enumerate(a):
        rst[row_idx, :len(row)] = row
    return rst


def find_alignment(seq1, seq2, del_cost, trans_factor):
    """
    Similar functionality with find_alignment_nc, but for single del_cost cost.
    :param np.ndarray seq1:
    :param np.ndarray seq2:
    :param float del_cost:
    :param float trans_factor:
    :return:
    """
    align_pairs, min_distance = \
        find_alignment_mc(seq1, seq2, np.array([del_cost]), trans_factor)
    return align_pairs[0], float(min_distance[0])


def float_equal(a, b):
    eps = 1e-4
    return (1-eps) < (a/b) < (1+eps)