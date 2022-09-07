import numpy as np
import pandas as pd
import torch


def _segment_index(x, chunklen, hoplen, last_frame_always_paddding=False):
    """Segment input x with chunklen, hoplen parameters. Return

    Args:
        x: input, time domain or feature domain (channels, time)
        chunklen:
        hoplen:
        last_frame_always_paddding: to decide if always padding for the last frame
    
    Return:
        segmented_indexes: [(begin_index, end_index), (begin_index, end_index), ...]
        segmented_pad_width: [(before, after), (before, after), ...]
    """
    x_len = x.shape[1]

    segmented_indexes = []
    segmented_pad_width = []
    if x_len < chunklen:
        begin_index = 0
        end_index = x_len
        pad_width_before = 0
        pad_width_after = chunklen - x_len
        segmented_indexes.append((begin_index, end_index))
        segmented_pad_width.append((pad_width_before, pad_width_after))
        return segmented_indexes, segmented_pad_width

    n_frames = 1 + (x_len - chunklen) // hoplen
    for n in range(n_frames):
        begin_index = n * hoplen
        end_index = n * hoplen + chunklen
        segmented_indexes.append((begin_index, end_index))
        pad_width_before = 0
        pad_width_after = 0
        segmented_pad_width.append((pad_width_before, pad_width_after))
    
    if (n_frames - 1) * hoplen + chunklen == x_len:
        return segmented_indexes, segmented_pad_width

    # the last frame
    if last_frame_always_paddding:
        begin_index = n_frames * hoplen
        end_index = x_len
        pad_width_before = 0
        pad_width_after = chunklen - (x_len - n_frames * hoplen)        
    else:
        if x_len - n_frames * hoplen >= chunklen // 2:
            begin_index = n_frames * hoplen
            end_index = x_len
            pad_width_before = 0
            pad_width_after = chunklen - (x_len - n_frames * hoplen)
        else:
            begin_index = x_len - chunklen
            end_index = x_len
            pad_width_before = 0
            pad_width_after = 0
    segmented_indexes.append((begin_index, end_index))
    segmented_pad_width.append((pad_width_before, pad_width_after))

    return segmented_indexes, segmented_pad_width


def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5: #polar coordinates 
            # _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4])])
        elif len(_words) == 6: # cartesian coordinates
            # _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
            _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4]), float(_words[5])])
        elif len(_words) == 4:
            _output_dict[_frame_ind].append([int(_words[1]), float(_words[2]), float(_words[3])])
    _fid.close()
    return _output_dict


def write_output_format_file(_output_format_file, _output_format_dict):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    _fid = open(_output_format_file, 'w')
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            # Write Cartesian format output. Since baseline does not estimate track count we use a fixed value.
            _fid.write('{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), int(_value[1]), int(_value[2])))
    _fid.close()


def to_metrics_format(label_dict, num_frames, label_resolution=0.1):
    """Collect class-wise sound event location information in segments of length 1s (according to DCASE2022) from reference dataset

    Reference:
        https://github.com/sharathadavanne/seld-dcase2022/blob/main/cls_feature_class.py
    Args:
        label_dict: Dictionary containing frame-wise sound event time and location information. Dcase format.
        num_frames: Total number of frames in the recording.
        label_resolution: Groundtruth label resolution.
    Output:
        output_dict: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth in degree, elevation in degree)
    """

    num_label_frames_1s = int(1 / label_resolution)
    num_blocks = int(np.ceil(num_frames / float(num_label_frames_1s)))
    output_dict = {x: {} for x in range(num_blocks)}
    for n_frame in range(0, num_frames, num_label_frames_1s):
        # Collect class-wise information for each block
        #    [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        n_block = n_frame // num_label_frames_1s
        loc_dict = {}
        for audio_frame in range(n_frame, n_frame + num_label_frames_1s):
            if audio_frame not in label_dict:
                continue            
            for value in label_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {}
                
                block_frame = audio_frame - n_frame
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append(value[1:])

        # Update the block wise details collected above in a global structure
        for n_class in loc_dict:
            if n_class not in output_dict[n_block]:
                output_dict[n_block][n_class] = []

            keys = [k for k in loc_dict[n_class]]
            values = [loc_dict[n_class][k] for k in loc_dict[n_class]]

            output_dict[n_block][n_class].append([keys, values])

    return output_dict

def track_to_dcase_format(sed_labels, doa_labels):
    """Convert sed and doa labels from track-wise output format to dcase output format

    Args:
        sed_labels: SED labels, (num_frames, num_tracks=3, logits_events=13 (number of classes))
        doa_labels: DOA labels, (num_frames, num_tracks=3, logits_degrees=2 (azi in radiance, ele in radiance))
    Output:
        output_dict: return a dict containing dcase output format
            output_dict[frame-containing-events] = [[class_index_1, azi_1 in degree, ele_1 in degree], [class_index_2, azi_2 in degree, ele_2 in degree]]
    """
    
    frame_size, num_tracks, num_classes= sed_labels.shape
    
    output_dict = {}
    for n_idx in range(frame_size):
        for n_track in range(num_tracks):
            class_index = list(np.where(sed_labels[n_idx, n_track, :])[0])
            assert len(class_index) <= 1, 'class_index should be smaller or equal to 1!!\n'
            if class_index:
                event_doa = [class_index[0], int(np.around(doa_labels[n_idx, n_track, 0] * 180 / np.pi)), \
                                            int(np.around(doa_labels[n_idx, n_track, 1] * 180 / np.pi))] # NOTE: this is in degree
                if n_idx not in output_dict:
                    output_dict[n_idx] = []
                output_dict[n_idx].append(event_doa)
    return output_dict

def convert_output_format_polar_to_cartesian(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:

                ele_rad = tmp_val[2]*np.pi/180.
                azi_rad = tmp_val[1]*np.pi/180

                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                out_dict[frame_cnt].append([tmp_val[0], x, y, z])
    return out_dict

def convert_output_format_cartesian_to_polar(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x, y, z = tmp_val[1], tmp_val[2], tmp_val[3]

                # in degrees
                azimuth = np.arctan2(y, x) * 180 / np.pi
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                r = np.sqrt(x**2 + y**2 + z**2)
                out_dict[frame_cnt].append([tmp_val[0], azimuth, elevation])
    return out_dict

def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


########################################
########## multi-accdoa
########################################
def get_multi_accdoa_labels(accdoa_in, nb_classes=13):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*13]
        nb_classes: scalar
    Return:
        sed:       [num_track, batch_size, frames, num_class=13]
        doa:       [num_track, batch_size, frames, num_axis*num_class=3*13]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 6*nb_classes:]
    sed = np.stack((sed0, sed1, sed2), axis=0)
    doa = np.stack((doa0, doa1, doa2), axis=0)

    return sed, doa
    
def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0

def multi_accdoa_to_dcase_format(sed_pred, doa_pred, threshold_unify=15,nb_classes=13):
    sed_pred0, sed_pred1, sed_pred2 = sed_pred
    doa_pred0, doa_pred1, doa_pred2 = doa_pred
    
    output_dict = {}
    for frame_cnt in range(sed_pred0.shape[0]):
        for class_cnt in range(sed_pred0.shape[1]):
            # determine whether track0 is similar to track1
            flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], \
                doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, threshold_unify, nb_classes)
            flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], \
                doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, threshold_unify, nb_classes)
            flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], \
                doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, threshold_unify, nb_classes)
            # unify or not unify according to flag
            if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                if sed_pred0[frame_cnt][class_cnt]>0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], \
                        doa_pred0[frame_cnt][class_cnt+nb_classes], doa_pred0[frame_cnt][class_cnt+2*nb_classes]])
                if sed_pred1[frame_cnt][class_cnt]>0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], \
                        doa_pred1[frame_cnt][class_cnt+nb_classes], doa_pred1[frame_cnt][class_cnt+2*nb_classes]])
                if sed_pred2[frame_cnt][class_cnt]>0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], \
                        doa_pred2[frame_cnt][class_cnt+nb_classes], doa_pred2[frame_cnt][class_cnt+2*nb_classes]])
            elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                if flag_0sim1:
                    if sed_pred2[frame_cnt][class_cnt]>0.5:
                        output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], \
                            doa_pred2[frame_cnt][class_cnt+nb_classes], doa_pred2[frame_cnt][class_cnt+2*nb_classes]])
                    doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], \
                        doa_pred_fc[class_cnt+nb_classes], doa_pred_fc[class_cnt+2*nb_classes]])
                elif flag_1sim2:
                    if sed_pred0[frame_cnt][class_cnt]>0.5:
                        output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], \
                            doa_pred0[frame_cnt][class_cnt+nb_classes], doa_pred0[frame_cnt][class_cnt+2*nb_classes]])
                    doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], \
                        doa_pred_fc[class_cnt+nb_classes], doa_pred_fc[class_cnt+2*nb_classes]])
                elif flag_2sim0:
                    if sed_pred1[frame_cnt][class_cnt]>0.5:
                        output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], \
                            doa_pred1[frame_cnt][class_cnt+nb_classes], doa_pred1[frame_cnt][class_cnt+2*nb_classes]])
                    doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], \
                        doa_pred_fc[class_cnt+nb_classes], doa_pred_fc[class_cnt+2*nb_classes]])
            elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], \
                    doa_pred_fc[class_cnt+nb_classes], doa_pred_fc[class_cnt+2*nb_classes]])
    return output_dict
