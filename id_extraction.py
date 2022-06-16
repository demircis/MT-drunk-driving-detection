import pandas as pd
import numpy as np
import cv2 as cv
import ffmpeg
import os
import math
import datetime


def map_to_digit(segments):
    zero = np.array([1, 1, 1, 0, 1, 1, 1])
    one = np.array([0, 0, 1, 0, 0, 1, 0])
    two = np.array([1, 0, 1, 1, 1, 0, 1])
    three = np.array([1, 0, 1, 1, 0, 1, 1])
    four = np.array([0, 1, 1, 1, 0, 1, 0])
    five = np.array([1, 1, 0, 1, 0, 1, 1])
    six = np.array([1, 1, 0, 1, 1, 1, 1])
    seven = np.array([1, 0, 1, 0, 0, 1, 0])
    eight = np.array([1, 1, 1, 1, 1, 1, 1])
    nine = np.array([1, 1, 1, 1, 0, 1, 0])

    if np.array_equal(segments, zero):
        return '0'
    elif np.array_equal(segments, one):
        return '1'
    elif np.array_equal(segments, two):
        return '2'
    elif np.array_equal(segments, three):
        return '3'
    elif np.array_equal(segments, four):
        return '4'
    elif np.array_equal(segments, five):
        return '5'
    elif np.array_equal(segments, six):
        return '6'
    elif np.array_equal(segments, seven):
        return '7'
    elif np.array_equal(segments, eight):
        return '8'
    elif np.array_equal(segments, nine):
        return '9'
    else:
        return None


def get_segment_states(digits, smaller_dimensions):
    segments_states = []
    for digit in digits:
        contours, _ = cv.findContours(digit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        full_contours = np.concatenate(contours)
        x, y, w, h = cv.boundingRect(full_contours)
        # extend bounding box to the left (fix for certain digits)
        standard_w = 6 if smaller_dimensions else 9
        if w < standard_w:
            x = x - (standard_w - w)
            w = standard_w
        digit_rect = digit[y:y + h, x:x + w]
        segment_w = 3
        segment_h = 3
        if smaller_dimensions:
            segment_w = 2
            segment_h = 2
        segment_h_center = 1 if smaller_dimensions else 2
        # top, top-left, top-right, center, bottom-left, bottom-right, bottom
        segments = [
            ((1, 0), (w-1, segment_h)),
            ((0, 1), (segment_w, h // 2)),
            ((w - segment_w, 1), (w, h // 2)),
            ((1, (h // 2) - segment_h_center) , (w-1, (h // 2) + segment_h_center)),
            ((0, h // 2), (segment_w, h-1)),
            ((w - segment_w, h // 2), (w, h-1)),
            ((1, h - segment_h), (w-1, h))
        ]
        segment_state = np.array([0] * len(segments))
        for j, ((xStart, yStart), (xEnd, yEnd)) in enumerate(segments):
            segment = digit_rect[yStart:yEnd, xStart:xEnd]
            total = cv.countNonZero(segment)
            area = (xEnd - xStart) * (yEnd - yStart)
            if area == 0:
                segment_state = np.array([0] * len(segments))
                break
            if total / float(area) >= 0.5:
                segment_state[j]= 1
        segments_states.append(segment_state)

    return segments_states


def extract_id(img, smaller_dimensions):
    imhsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(imhsv, (0, 50, 0), (179, 255, 255))
    img_no_artifacts = cv.bitwise_and(img, img, mask=mask)
    imgray = cv.cvtColor(img_no_artifacts, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(imgray, 85, 255, cv.THRESH_BINARY) if smaller_dimensions else cv.threshold(imgray, 115, 255, cv.THRESH_BINARY)
    digits = np.array_split(thresh, 5, axis=1)
    extracted_id = ''
    for segment_state in get_segment_states(digits, smaller_dimensions):
        extracted_id += map_to_digit(segment_state)
    extracted_id = int(extracted_id) if extracted_id != '' else None
    return extracted_id


def get_ids_for_indices(path_ids, segment_ids, cropped_video, dimensions, timestamps, indices):
    cap = cv.VideoCapture(cropped_video)
    for ind, timestamp_ms in zip(indices, timestamps[indices]):
        is_set = cap.set(cv.CAP_PROP_POS_MSEC, timestamp_ms)
        success, frame = cap.read()
        if is_set and success:
            end = dimensions['path_frame']
            start = dimensions['segment_frame']
            path_img = frame[:end, :, :]
            segment_img = frame[start:, :, :]
            try:
                path_ids[ind] = extract_id(path_img, True) if (end == 15 and start == 20) else extract_id(path_img, False)
                segment_ids[ind] = extract_id(segment_img, True) if (end == 15 and start == 20) else extract_id(segment_img, False)
            except TypeError:
                subject_folder = cropped_video.split('/')[4]
                if not os.path.exists('out/{}'.format(subject_folder)):
                    os.makedirs('out/{}'.format(subject_folder))
                cv.imwrite('out/{}/error_path_{}.jpg'.format(subject_folder, timestamp_ms), path_img)
                cv.imwrite('out/{}/error_segment_{}.jpg'.format(subject_folder, timestamp_ms), segment_img)
                f = open('out/error_digits.txt', 'a')
                f.write('video: {}, timestamp (ms): {}, index: {}\n'.format(cropped_video, timestamp_ms, ind))
                f.close()
        else:
            print('could not get frame at timestamp')
            break

    cap.release()
    return path_ids, segment_ids


def crop_video(video, crop_dimensions):
    (
    ffmpeg
    .input(video)
    .crop(crop_dimensions['x'], crop_dimensions['y'], crop_dimensions['height'], crop_dimensions['width'])
    .output(video[:-4] + '_cropped.flv', vcodec='libx264', acodec='copy', preset='ultrafast')
    .run()
    )


def get_path_and_segment_ids(video, dimensions, data_timestamps, video_timestamp, data_freq):
    can_data_timestamps_ms = ((data_timestamps - video_timestamp) / datetime.timedelta(milliseconds=1)).to_numpy()
    nr_timestamps = len(can_data_timestamps_ms)
    path_ids = np.array([np.nan] * nr_timestamps)
    segment_ids = np.array([np.nan] * nr_timestamps)

    sampling_indices = np.arange(0, nr_timestamps, data_freq * 5)
    last_index = nr_timestamps-1
    if sampling_indices[-1] != last_index:
        sampling_indices = np.append(sampling_indices, last_index)

    cropped_video = video[:-4] + '_cropped.flv'
    assert(cropped_video != video)

    if not os.path.exists(cropped_video):
        crop_video(video, dimensions)

    path_ids, segment_ids = get_ids_for_indices(path_ids, segment_ids, cropped_video, dimensions, can_data_timestamps_ms, sampling_indices)

    prev_none = -1
    while len(segment_ids[np.isnan(segment_ids)]) != 0:
        new_indices = []
        for i in range(len(sampling_indices)-1):
            left = sampling_indices[i]
            right = sampling_indices[i+1]
            if path_ids[left] == path_ids[right]:
                path_ids[left:right] = path_ids[left]

            if segment_ids[left] == segment_ids[right]:
                segment_ids[left:right] = segment_ids[left]
            else:
                new_indices.append(left+(right-left)//2)

        new_indices = np.array(new_indices, dtype=np.int64)
        sampling_indices = np.concatenate((sampling_indices, new_indices))
        sampling_indices = np.sort(sampling_indices)
        path_ids, segment_ids = get_ids_for_indices(path_ids, segment_ids, cropped_video, dimensions, can_data_timestamps_ms, new_indices)
        sampling_indices = sampling_indices[np.invert(np.isnan(segment_ids[sampling_indices]))]

        if len(segment_ids[np.isnan(segment_ids)]) == prev_none:
            break
        prev_none = len(segment_ids[np.isnan(segment_ids)])

    return path_ids, segment_ids


def find_closest_segment(lanes_df, target_xpos, target_ypos):
    samples = 10
    distances = [[math.sqrt((target_xpos - x)**2 + (target_ypos - y)**2) for x, y in zip(np.linspace(start_x, end_x, samples), np.linspace(start_y, end_y, samples))]
        for start_x, start_y, end_x, end_y in zip(lanes_df['StartPos_x_segment'], lanes_df['StartPos_y_segment'], lanes_df['StartPos_x_segment'], lanes_df['EndPos_y_segment'])]
    min_per_segment = np.min(distances, axis=1)
    min_idx = np.argmin(min_per_segment)
    return lanes_df.iloc[min_idx]['segment_id']


def get_distance_based_path_and_segment_ids(lanes, xpositions, ypositions):
    path_ids = []
    segment_ids = []
    path_id_order = lanes['path_id'].unique()
    nr_paths = len(path_id_order)
    i = 0
    prev_path_id = path_id_order[i]
    for xpos, ypos in zip(xpositions, ypositions):
        lanes_subset = None
        if i == nr_paths-1:
            lanes_subset = lanes.loc[lanes['path_id'] == path_id_order[i]]
        else:
            lanes_subset = lanes.loc[(lanes['path_id'] == path_id_order[i]) | (lanes['path_id'] == path_id_order[i+1])]
        segment_id = find_closest_segment(lanes_subset, xpos, ypos)
        segment_ids.append(segment_id)
        path_id = lanes.loc[lanes['segment_id'] == segment_id, 'path_id'].to_numpy()[0]
        path_ids.append(path_id)
        if path_id != prev_path_id:
            i += 1
        prev_path_id = path_id
    return np.array(path_ids), np.array(segment_ids)