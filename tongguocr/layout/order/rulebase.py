from typing import List, Tuple

import cv2
import numpy as np

from .polygon_utils import (compose_polygon, find_center_line, get_line_angle,
                            rotate_line, shink_poly)


def load_boxes(coordinates_batch: List[np.ndarray],
               text_batch: List[str],
               img_shape: Tuple[int, int],
               minimal_rotate_angle: float = 3.):
    """Load boxes from txt file. Return a list of boxes and the average height
    and average width of the boxes.

    Args:
        anno_path (str): path to txt file
        img_shape (Tuple[int, int]): shape of image in (height, width)
        minimal_rotate_angle (float): Minimal rotation angle in degree.

    Returns:
        boxes (List(dict)): list of dicts. Each dict contains the following
            keys:
                - 'coordinates' (np.ndarray): List of points in shape (N, 2).
                - 'centerline' (np.ndarray): Center line of the box in shape
                    (4, ).
                - 'minirect' (np.ndarray): The mimimum circumscribed rectangle
                    in shape (4, 2). This minirect is actually the minimum
                    rectangle due to design drawbacks. This one is used to
                    find the first box.
                - 'real_minirect' (np.ndarray): The mimimum circumscribed
                    rectangle in shape (4, 2). This one is used to find the
                    order.
                - 'shrunk_minirect' (np.ndarray): The shrunk mimimum
                    circumscribed rectangle in shape (4, 2). This one is used
                    to segment.
                - 'height' (float): Approximate height of the box.
                - 'width' (float): Approximate width of the box.
                - 'text' (str): Text of the box.
                - 'index' (int): Index of the box in the original list.
        rotated (bool): Whether the boxes are rotated.
        distored (bool): Whether the boxes are distored.
    """
    boxes = []
    angles = []
    rotated = False
    distored = False
    for i, (coordinates,
            text) in enumerate(zip(coordinates_batch, text_batch)):
        coordinates = np.array(coordinates, dtype=np.float32).reshape(-1, 2)
        center_line, width, height = find_center_line(coordinates)
        angle = get_line_angle(center_line)
        if abs(angle) > minimal_rotate_angle:
            distored = True
        angles.append(angle)
        x, y, w, h = cv2.boundingRect(coordinates)
        minirect = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                            dtype=np.float32)
        # clip minirect to image
        minirect[:, 0] = np.clip(minirect[:, 0], 0, img_shape[1] - 1)
        minirect[:, 1] = np.clip(minirect[:, 1], 0, img_shape[0] - 1)
        # shrunk_minirect is used for segment
        shrunk_minirect = shink_poly(minirect, height, 0.95)
        # real minirect
        real_minirect = cv2.boxPoints(cv2.minAreaRect(coordinates))
        real_minirect = np.intp(real_minirect)
        boxes.append(
            dict(
                coordinates=coordinates,
                centerline=center_line,
                minirect=minirect,
                shrunk_minirect=shrunk_minirect,
                real_minirect=real_minirect,
                height=height,
                width=width,
                text=text,
                index=i))

    # find avearge angle of boxes
    avg_angle = np.mean(np.array(angles, np.float32))
    if abs(avg_angle) > minimal_rotate_angle:
        rotated = True
        distored = False
        for box in boxes:
            centerline = rotate_line(img_shape, box['centerline'], avg_angle)
            box['centerline'] = centerline
            box['minirect'] = compose_polygon(centerline, box['width'])
            box['shrunk_minirect'] = shink_poly(box['minirect'], box['height'],
                                                0.95)

    return boxes, rotated, distored


def sort_boxes(boxes: List[dict], order: List[int]):
    """Sort boxes given the order.

    Args:
        boxes (List[dict]): List of dicts. Each dict contains the following
            keys:
                - 'coordinates' (np.ndarray): List of points in shape (N, 2).
                - 'centerline' (np.ndarray): Center line of the box in shape
                    (4, ).
                - 'minirect' (np.ndarray): The mimimum circumscribed rectangle
                    in shape (4, 2). This minirect is actually the minimum
                    rectangle due to design drawbacks. This one is used to
                    find the first box.
                - 'real_minirect' (np.ndarray): The mimimum circumscribed
                    rectangle in shape (4, 2). This one is used to find the
                    order.
                - 'shrunk_minirect' (np.ndarray): The shrunk mimimum
                    circumscribed rectangle in shape (4, 2). This one is used
                    to segment.
                - 'height' (float): Approximate height of the box.
                - 'width' (float): Approximate width of the box.
                - 'text' (str): Text of the box.
                - 'index' (int): Index of the box in the original list.
        order (List[int]): The order of boxes.

    Returns:
        List[List[int]]: Sorted boxes.
    """
    return [
        boxes[i]['coordinates'].reshape(-1).tolist() + [boxes[i]['text']]
        for i in order
    ]


def is_in_segment(segment: Tuple[int, int],
                  y_coords: Tuple[int, int],
                  thresh: float = 0.85):
    """Check whether the mojarity of a box is in a segment.

    Args:
        segment (Tuple[int, int]): The segment in the form of (start, end).
        y_coords (Tuple[int, int]): The y coordinates of the box in the form of
            (top, bottom).
        thresh (float): The threshold of the mojarity.

    Returns:
        bool: Whether the box is in the segment.
    """
    if y_coords[0] >= segment[1] or y_coords[1] <= segment[0]:
        return False
    if y_coords[0] >= segment[0] and y_coords[1] <= segment[1]:
        return True
    intersection = segment[1] - y_coords[0]
    if (intersection / (y_coords[1] - y_coords[0])) > thresh:
        return True
    else:
        return False


def segment_raw(boxes: List[dict], image_shape: Tuple[int, int], rotated: bool,
                distored: bool):
    """Segment boxes in raw direction. The boxes may be located in different
    segments from above and below, and the coordinates are projected onto the
    y-axis for segmentation.

    Args:
        polys : List(dict): list of dicts. Each dict contains the following
            boxes:
                - 'coordinates' (np.ndarray): List of points in shape (N, 2).
                - 'centerline' (np.ndarray): Center line of the box in shape
                    (4, ).
                - 'minirect' (np.ndarray): THe mimimum circumscribed rectangle
                    in shape (4, 2).
                - 'real_minirect' (np.ndarray): The mimimum circumscribed
                    rectangle in shape (4, 2). This one is used to find the
                    order.
                - 'shrunk_minirect' (np.ndarray): The shrunk mimimum
                    circumscribed rectangle in shape (4, 2). This one is used
                    to segment.
                - 'height' (float): Approximate height of the box.
                - 'width' (float): Approximate width of the box.
                - 'index' (int): Index of the box in the original list.
        image_shape (Tuple[int, int]): shape of image in (height, width)
        rotated (bool): Whether the boxes are rotated.
        distored (bool): Whether the boxes are distored.

    Returns:
        segment_boxes: (List[dict]): List of dicts in structure as follows:
            [
                {
                    'avg_height': float,
                    'avg_width': float,
                    'rotated': bool,
                    'distored': bool,
                    'boxes': [
                        {
                            'coordinates': np.ndarray,
                            'centerline': np.ndarray,
                            'minirect': np.ndarray,
                            'real_minirect': np.ndarray,
                            'shrunk_minirect': np.ndarray,
                            'height': float,
                            'width': float,
                            'index': int
                        },
                        ...
                    ]
                }
                ...
            ]
    """
    projection_regions = np.zeros(image_shape[0], dtype=np.int32)
    candidate_indexs = [
    ]  # use to find if there are boxes that are not in any segment
    for box in boxes:
        coordinates = box['shrunk_minirect']

        candidate_indexs.append(box['index'])
        # find the minimal y coordinate
        min_y = np.min(coordinates[:, 1])
        # find the maximal y coordinate
        max_y = np.max(coordinates[:, 1])
        projection_regions[int(min_y):int(max_y)] = 1
    # segment projection regions based on the y-axis
    segments = []
    start = 0
    continue_to_be_1 = False
    for i in range(1, len(projection_regions)):
        if projection_regions[i] == 1 and projection_regions[i - 1] == 0:
            start = i
            continue_to_be_1 = True
        if projection_regions[i] == 0 and projection_regions[i - 1] == 1:
            segments.append([start, i])
            continue_to_be_1 = False
    if continue_to_be_1 is True:
        segments.append([start, image_shape[0] - 1])
    if sum(projection_regions) == image_shape[0]:
        segments.append([0, image_shape[0] - 1])
    # segment boxes based on the segments
    # each segment corresponds to a list of boxes
    segment_boxes = []

    for segment in segments:
        group_boxes = []
        segment_heights = []
        segment_widths = []
        for i, box in enumerate(boxes):
            coordinates = box['shrunk_minirect']
            # find the minimal y coordinate
            min_y = np.min(coordinates[:, 1])
            # find the maximal y coordinate
            max_y = np.max(coordinates[:, 1])
            # if mojarity of the box is in inside segment, then the box is
            # in the segment
            if is_in_segment(segment, (min_y, max_y)):
                candidate_indexs.remove(box['index'])
                group_boxes.append(box)
                segment_heights.append(box['height'])
                segment_widths.append(box['width'])

        avg_height = np.mean(np.array(
            segment_heights, np.float32)) if len(segment_heights) > 0 else 0
        avg_width = np.mean(np.array(
            segment_widths, np.float32)) if len(segment_widths) > 0 else 0
        if avg_height == 0 or avg_width == 0:
            continue
        segment_boxes.append(
            dict(
                avg_height=avg_height,
                avg_width=avg_width,
                rotated=rotated,
                distored=distored,
                boxes=group_boxes))
    if len(candidate_indexs) > 0:
        print(f'Warning: {len(candidate_indexs)} boxes are not in any segment')
    return segment_boxes


def is_line_adjcent(main_line: np.ndarray,
                    sub_line: np.ndarray,
                    overlap_thresh: float = 0.5,
                    gap_thresh: float = 0.5):
    """Given two lines in shape (2,2), check if there are adjcent. If two boxes
    are adjcent, there projection on x axis must have a large overlap, and the
    gap between their height should be small

    Args:
        main_line (np.ndarray): The main line in shape (2,2).
        sub_line (np.ndarray): The sub line in shape (2,2).
        overlap_thresh (float): The overlap threshold.
        gap_thresh (float): The gap threshold.

    Returns:
        bool: Whether the two lines are adjcent.
    """
    # find the projection of the two lines on x axis
    main_line_x = main_line[:, 0]
    sub_line_x = sub_line[:, 0]
    # find the projection of the two lines on y axis
    main_line_y = main_line[:, 1]
    sub_line_y = sub_line[:, 1]
    # find the overlap of the two lines on x axis
    intersection = min(main_line_x[1], sub_line_x[1]) - max(
        main_line_x[0], sub_line_x[0])
    # find the gap between the two lines on y axis
    gap = abs(np.mean(main_line_y) - np.mean(sub_line_y))
    # if the overlap is large and the gap is small, then the two lines are
    main_length = main_line_x[1] - main_line_x[0]
    sub_length = sub_line_x[1] - sub_line_x[0]
    refer_long_length = max(main_length, sub_length)
    refer_short_length = min(main_length, sub_length)
    if intersection / refer_short_length > overlap_thresh:
        if gap <= gap_thresh * refer_long_length:
            return True
    else:
        return False


def find_box_relation(main_box: dict, sub_box: dict, avg_width: float):
    """Given two boxes, check what is sub_box to main_box

    Args:
        box1 (dict): The first box.
        box2 (dict): The second box.
        avg_width (float): The average width of the boxes in the segment.

    Returns:
        (str): Relationship in ['top', 'down', 'none']
    """
    # main_box
    main_coordinates = main_box['real_minirect']
    sub_coordinates = sub_box['real_minirect']

    min_ys = np.argsort(main_coordinates[:, 1])[:2]
    main_top_line = main_coordinates[min_ys]
    # sort main_top_line by x coordinate
    main_top_line = main_top_line[np.argsort(main_top_line[:, 0])]

    max_ys = np.argsort(main_coordinates[:, 1])[2:]
    main_down_line = main_coordinates[max_ys]
    # sort main_down_line by x coordinate
    main_down_line = main_down_line[np.argsort(main_down_line[:, 0])]

    # sub_box
    min_ys = np.argsort(sub_coordinates[:, 1])[:2]
    sub_top_line = sub_coordinates[min_ys]
    # sort sub_top_line by x coordinate
    sub_top_line = sub_top_line[np.argsort(sub_top_line[:, 0])]

    max_ys = np.argsort(sub_coordinates[:, 1])[2:]
    sub_down_line = sub_coordinates[max_ys]
    # sort sub_down_line by x coordinate
    sub_down_line = sub_down_line[np.argsort(sub_down_line[:, 0])]

    # check if sub_box is top of main_box
    if is_line_adjcent(main_top_line, sub_down_line):
        return 'top'
    # check if sub_box is down of main_box
    if is_line_adjcent(main_down_line, sub_top_line):
        return 'down'
    return 'none'


def is_in_groups(group_boxes: dict, query_box: dict):
    """Given a group of boxes, check if the box is in the group.

    Args:
        group_boxes (dict): The group of boxes.
        query_box (dict): The box.

    Returns:
        (str): If the box is in the group, return its corresponding index in
            the group. Otherwise, return 'none'.
    """
    for group_index, group in group_boxes.items():
        if query_box['index'] in group['box_indexes']:
            return group_index
    return 'none'


def segment_col(segment: dict):
    """Given a set of boxes, segment them in colum direction
    Args:
        segment (dict): A dict of segment in structure as follows:
            {
                    'avg_height': float,
                    'avg_width': float,
                    'rotated': bool,
                    'distored': bool,
                    'boxes': [
                        {
                            'coordinates': np.ndarray,
                            'centerline': np.ndarray,
                            'minirect': np.ndarray,
                            'real_minirect': np.ndarray,
                            'shrunk_minirect': np.ndarray,
                            'height': float,
                            'width': float,
                            'index': int
                        },
                        ...
                    ]
                }
    Return:
        groups (dict): A dict of groups in structure as follows:
            {
                '1' : {
                        'box_indexes': [1, 2, 3],
                        'boxes': [...]
                    },
                ...
            }
    """
    avg_width = segment['avg_width']
    boxes = segment['boxes']
    groups = {}  # groups: {
    #            '1':{'box_indexes': [1, 2, 3], 'boxes': [...]},
    # }}
    num_groups = 0
    # sort the boxes by their centerline's x coordinate
    pre_sort_boxes = sorted(boxes, key=lambda x: x['centerline'][0])
    for i, pre_box_query in enumerate(pre_sort_boxes):
        in_group = False
        top_found = False
        down_found = False
        for j, pre_box_candidate in enumerate(pre_sort_boxes):
            if top_found and down_found:
                break
            if i == j:
                continue
            # find the relationship between the two boxes
            relation = find_box_relation(pre_box_query, pre_box_candidate,
                                         avg_width)
            if relation == 'none':
                continue
            # if the two boxes are adjcent
            else:
                if relation == 'top':
                    top_found = True
                if relation == 'down':
                    down_found = True
                in_group = True
                query_index = is_in_groups(groups, pre_box_query)
                candidate_index = is_in_groups(groups, pre_box_candidate)
                if query_index == candidate_index and query_index != 'none':
                    # query and candidate are already in the same group
                    continue
                # case1: if query is in group and candidate in group, then
                # merge the two groups. The way to merge is to move all the
                # boxes in the candidate group to the query group.
                if query_index != 'none' and candidate_index != 'none':
                    query_boxes_indexes = groups[query_index]['box_indexes']
                    candidate_boxes_indexes = groups[candidate_index][
                        'box_indexes']
                    # find the difference between the two groups
                    difference = set(candidate_boxes_indexes).difference(
                        query_boxes_indexes)
                    # move the difference boxes to the query group
                    for index in difference:
                        groups[query_index]['box_indexes'].append(index)
                    for candidate_box in groups[candidate_index]['boxes']:
                        if candidate_box['index'] in difference:
                            groups[query_index]['boxes'].append(candidate_box)
                    # remove the candidate group
                    del groups[candidate_index]
                # case2: if query is in group and candidate not in group, then
                # move the candidate to the query group.
                elif query_index != 'none' and candidate_index == 'none':
                    groups[query_index]['box_indexes'].append(
                        pre_box_candidate['index'])
                    groups[query_index]['boxes'].append(pre_box_candidate)

                # case3: if query not in group and candidate in group, then
                # move the query to the candidate group.
                elif query_index == 'none' and candidate_index != 'none':
                    groups[candidate_index]['box_indexes'].append(
                        pre_box_query['index'])
                    groups[candidate_index]['boxes'].append(pre_box_query)

                # case4: if query not in group and candidate not in group, then
                # create a new group.
                elif query_index == 'none' and candidate_index == 'none':
                    groups[str(num_groups)] = dict(
                        box_indexes=[
                            pre_box_query['index'], pre_box_candidate['index']
                        ],
                        boxes=[pre_box_query, pre_box_candidate])
                    num_groups += 1
        # if the box is not in any group, create a new group.
        if not in_group:
            groups[str(num_groups)] = dict(
                box_indexes=[pre_box_query['index']], boxes=[pre_box_query])
            num_groups += 1
    return groups


def find_start_box(segment: dict, num_candidates: int = 3):
    """Given a set of boxes, find the start box.
    Args:
        segment (dict): A dict of segment in structure as follows:
            {
                    'avg_height': float,
                    'avg_width': float,
                    'rotated': bool,
                    'distored': bool,
                    'boxes': [
                        {
                            'coordinates': np.ndarray,
                            'centerline': np.ndarray,
                            'minirect': np.ndarray,
                            'real_minirect': np.ndarray,
                            'shrunk_minirect': np.ndarray,
                            'height': float,
                            'width': float,
                            'index': int
                        },
                        ...
                    ]
                }
        num_candidates (int): number of candidates to be considered

    Returns:
        start_index (int): The index of the starting box.
    """
    # rule: Firstly, find num_candidates boxes that has the largest x
    # coordinate. Then, consider two cases:
    # 1. If the boxes are in the same line (the span on x axis of their
    # minimum circumscribed rectangle will be huge), the box with the largest x
    # coordinate is the start box.
    # 2. If the boxes are not in the same colume (the span on y axis of their
    # minimum circumscribed rectangle will be huge), the box with the smallest
    # y coordinate is the start box.

    top_rights = np.array([box['minirect'][1] for box in segment['boxes']],
                          dtype=np.float32)
    # find the top num_candidates boxes with the largest x coordinate, also
    # get their corresponding indexs
    top_candidates_index = np.argsort(top_rights[:, 0])[-num_candidates:]
    top_candidates = top_rights[np.argsort(top_rights[:, 0])[-num_candidates:]]

    # case 1: the boxes are in the same line
    if np.max(top_candidates[:, 0]) - np.min(top_candidates[:, 0]) > (
            num_candidates // 2) * segment['avg_width']:
        start_index = top_candidates_index[np.argmax(top_candidates[:, 0])]
    # case 2: the boxes are not in the same line
    else:
        start_index = top_candidates_index[np.argmin(top_candidates[:, 1])]
    return segment['boxes'][start_index]['index']


def reconstruct_line(group: List[dict]):
    """Given a group of boxes, find the top line

    Args:
        group (List[dict]): A list of boxes in structure as follows:
            [
                {
                    'coordinates': np.ndarray,
                    'centerline': np.ndarray,
                    'minirect': np.ndarray,
                    'real_minirect': np.ndarray,
                    'shrunk_minirect': np.ndarray,
                    'height': float,
                    'width': float,
                    'index': int
                }
            ]
    Returns:
        line (np.ndarray): The top line of the group.
    """
    width = np.max(np.array([box['width'] for box in group]))
    if len(group) == 1:
        top_right = group[0]['minirect'][1]
        line = [top_right[0] - width, top_right[1], top_right[0], top_right[1]]
    else:
        boxes = np.array([box['minirect'][1] for box in group])
        # find box_nums -1 boxes with the smallest y coordinate, also get their
        # corresponding indexs
        top_candidates = boxes[np.argsort(boxes[:, 1])[0]]
        line = [
            top_candidates[0] - width, top_candidates[1], top_candidates[0],
            top_candidates[1]
        ]
    return np.array(line)


def projection_boxes(boxes: List[dict]):
    """Projection a list of boxes to y axis

    Args:
        boxes (List[dict]): A list of boxes in structure as follows:
            [
                {
                    'coordinates': np.ndarray,
                    'centerline': np.ndarray,
                    'minirect': np.ndarray,
                    'real_minirect': np.ndarray,
                    'shrunk_minirect': np.ndarray,
                    'height': float,
                    'width': float,
                    'index': int
                }
            ]
    Returns:
        projections (np.ndarray): The projection of the boxes to y axis.
    """
    projections_y = []
    for box in boxes:
        projections_y += box['coordinates'][:, 1].tolist()
    return np.array(projections_y)


def inner_sort(group: dict):
    """Sort a single group based on ruls

    Args:
        group (dict): A dict in structure as follows:
            {
                'box_indexes': [1, 2, 3],
                'boxes': 
                    [
                        {
                            'coordinates': np.ndarray,
                            'centerline': np.ndarray,
                            'minirect': np.ndarray,
                            'real_minirect': np.ndarray,
                            'shrunk_minirect': np.ndarray,
                            'height': float,
                            'width': float,
                            'index': int
                        }
                        ...
                    ]
            }
    Returns:
        indexs (List[int]): The indexs of the sorted boxes.
    """
    # if there is only one box in grop, return the index directly
    if len(group['boxes']) == 1:
        return [group['boxes'][0]['index']]

    # if there are multiple boxes in group, sort them by their y coordinates.
    boxes = np.array([box['minirect'][1] for box in group['boxes']])
    
    indexs = np.argsort(boxes[:, 1])
    # print(group['boxes'])
    all_box=group['boxes']
    count=0
    same_line=[]
    sl_count=0
    for i in indexs:
        # print(all_box[i]['minirect'])
        # print(all_box[i]['minirect'][1],all_box[i]['minirect'][2])
        # print("boxes：")
        # print(boxes[i])
        box_height=all_box[i]['minirect'][2][1]-all_box[i]['minirect'][1][1]
        if(count<(len(indexs)-1)):
            
            boxes_distance=abs(boxes[indexs[count]][1]-boxes[indexs[count+1]][1])
            
            # print("bh:")
            # print(box_height)
            # print("bd:")
            # print(boxes_distance)
            if(box_height*0.8>boxes_distance):

                # print("位于同一行")
                # print("--------")
                # print(boxes[indexs[count]][0])
                # print(boxes[indexs[count+1]][0])
                # print("--------")
                if(boxes[indexs[count]][0]<boxes[indexs[count+1]][0]):
                    temp=indexs[count]
                    indexs[count]=indexs[count+1]
                    indexs[count+1]=temp
                    # print(indexs)
                    # print(len(indexs)) 
        # if(count<(len(indexs)-2)):
            count=count+1
            # print("count:")
            # print(count)

    # print(boxes)
    # print(indexs)
    return [group['boxes'][i]['index'] for i in indexs]


def find_groups_order(groups: dict):
    """Given a segment of boxes, find their reading order by rule.

    Args:
        groups (dict): A dict of groups in structure as follows:
            {
                '1' : {
                        'box_indexes': [1, 2, 3],
                        'boxes': [...]
                    },
                ...
            }
        Returns:
            List[int]: The index of the order.
    """
    # simlplest case: Each group has only one box, them sort the boxes by their
    # x coordinate.
    single = True
    for key in groups:
        if len(groups[key]['box_indexes']) > 1:
            single = False
            break
    if single:
        boxes = []
        for key in groups:
            boxes.append(groups[key]['boxes'][0])
        sorted_boxes = sorted(
            boxes, key=lambda x: x['real_minirect'][1][0], reverse=True)
        return [box['index'] for box in sorted_boxes]

    # Step1: We need to represent each group as a line, so that we can
    # sort and merge them. To do this, we need to find its top right point and
    # the everage width. Then, we reconstruct the line.
    for key in groups:
        groups[key]['top_line'] = reconstruct_line(groups[key]['boxes'])
    # print(groups)
    # Step2: TOTO: Merge some groups if there line have a huge overlap at
    # y axis.

    # Step3: Sort the groups by their top_line's top right point, from large
    # to small.
    top_rigth_xs = np.array([groups[key]['top_line'][2] for key in groups])
    # print("-------------------------------------------------------------------------")
    # print(top_rigth_xs)
    sorted_group_indexes = np.argsort(top_rigth_xs)[::-1]
    # print(sorted_group_indexes)
    # print("-------------------------------------------------------------------------")
    sorted_groups = [
        groups[list(groups.keys())[key]] for key in sorted_group_indexes
    ]
    # print(sorted_groups)
    # Step4: Find the order for each group
    sorted_groups = [inner_sort(group) for group in sorted_groups]
    # print("-------------------------------------------------------------------------")
    # print(sorted_groups)
    indexes = []
    for group in sorted_groups:
        indexes.extend(group)
    return indexes


def get_order(coordinates_batch: List[np.ndarray], text_batch: List[str],
              img_shape: Tuple[int, int]):
    """Main function.

    Args:
        img_path (str): The path of the image.
        csv_path (str): The path of the csv file.
        out_path (str): The path of the output.
    """
    boxes, rotated, distored = load_boxes(coordinates_batch, text_batch,
                                          img_shape)
    raw_segments = segment_raw(boxes, img_shape, rotated, distored)
    segment_orders = []
    for _, raw_segment in enumerate(raw_segments):
        groups = segment_col(raw_segment) #groups按照上下分成的组
        segment_orders += find_groups_order(groups)
    # sort boxes
    # print("groups")
    # print(groups)
    # print("segment_orders")
    # print(segment_orders)
    # print("boxes")
    # print(boxes)
    sorted_boxes = sort_boxes(boxes, segment_orders)
    # print("----------------------------------------------------")
    # print(sorted_boxes)
    return sorted_boxes
