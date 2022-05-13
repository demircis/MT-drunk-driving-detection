import pandas as pd
import re
import math

def parse_route_details(route_details, scenario_file_lines, idx, scenario):
    route_pattern = re.compile('Part\[MainTarget]\.(PathNr|Route)[\s]*:= ([0-9]+);')
    match = route_pattern.search(scenario_file_lines[idx])
    if match:
        path_id = match.group(2)
        route_details.append({'path_id': int(path_id), 'scenario': scenario})
    return route_details


def parse_path_details(path_details, road_file_lines, idx, scenario):
    path_pattern = re.compile('Path Nr.[\s]*([0-9]+)')
    path_segments_pattern = re.compile('[\s]*([0-9]+)')
    path_match = path_pattern.search(road_file_lines[idx])
    path_segments = path_segments_pattern.findall(road_file_lines[idx+3])
    if path_match:
        path_id = int(path_match.group(1))
        for s_id in path_segments:
            path_details.append({'path_id': path_id, 'scenario': scenario, 'segment_id': int(s_id)})
    return path_details


def parse_segment_details(segment_details, road_file_lines, idx, scenario):
    segment_pattern = re.compile('Segment Nr.[\s]*([0-9]+)')
    dlanes_pattern = re.compile('NrDLanes[\s]*([0-9]+)')
    lane_pattern = re.compile('^Lane[\s]*([0-9]+)')
    position_pattern = re.compile('(StartPos|EndPos)[\s]+([\-0-9]+.[0-9]+)[\s]+([\-0-9]+.[0-9]+)')
    angle_pattern = re.compile('(StartAng|EndAng)[\s]+([\-0-9]+.[0-9]+)')
    segment_id = None
    segment_start_pos_x = None
    segment_start_pos_y = None
    segment_end_pos_x = None
    segment_end_pos_y = None
    angle = 0
    segment_match = segment_pattern.search(road_file_lines[idx])
    if segment_match:
        segment_id = int(segment_match.group(1))
    segment_pos_match = position_pattern.findall(road_file_lines[idx+10])
    for s, x, y in segment_pos_match:
        if s == 'StartPos':
            segment_start_pos_x = float(x)
            segment_start_pos_y = float(y)
        if s == 'EndPos':
            segment_end_pos_x = float(x)
            segment_end_pos_y = float(y)
    angle_match = angle_pattern.findall(road_file_lines[idx+11])
    for s, matched_angle in angle_match:
        if s == 'StartAng':
            angle = math.degrees(float(matched_angle))
        elif s == 'EndAng':
            angle -= math.degrees(float(matched_angle))
    angle = abs(angle)
    offset = None
    for i, line in enumerate(road_file_lines[idx:idx+20]):
        if 'NrDLanes' in line:
            offset = i
            break
    if 'NrDLanes' in road_file_lines[idx+offset]:
        dlanes_match = dlanes_pattern.search(road_file_lines[idx+offset])
        if dlanes_match:
            nr_driving_lanes = int(dlanes_match.group(1))
        for i in range(nr_driving_lanes):
            base_idx = idx+i*9
            lane_nr_match = lane_pattern.search(road_file_lines[base_idx+offset+2])
            lane_id = None
            if lane_nr_match:
                lane_id = int(lane_nr_match.group(1))
                segment_details.append({'segment_id': segment_id, 'scenario': scenario, 'lane_id': lane_id, 'StartPos_x': segment_start_pos_x,
                                            'StartPos_y': segment_start_pos_y, 'EndPos_x': segment_end_pos_x, 
                                            'EndPos_y': segment_end_pos_y, 'angle': angle})
    return segment_details


def parse_lane_details(lane_details, road_file_lines, idx, scenario):
    dlanes_pattern = re.compile('NrDLanes[\s]*([0-9]+)')
    lane_pattern = re.compile('^Lane[\s]*([0-9]+)')
    position_pattern = re.compile('(StartPos|EndPos)[\s]+([\-0-9]+.[0-9]+)[\s]+([\-0-9]+.[0-9]+)')
    lane_width_pattern = re.compile('(LaneWidth|LeftEdgeLineWidth|RightEdgeLineWidth)[\s-]*([0-9]+.[0-9]+)')
    dlanes_match = dlanes_pattern.search(road_file_lines[idx])
    if dlanes_match:
        nr_driving_lanes = int(dlanes_match.group(1))
    for i in range(nr_driving_lanes):
        base_idx = idx+2+i*9
        lane_nr_match = lane_pattern.search(road_file_lines[base_idx])
        lane_id = None
        lane_start_pos_x = None
        lane_start_pos_y = None
        lane_end_pos_x = None
        lane_end_pos_y = None
        if lane_nr_match:
            lane_id = int(lane_nr_match.group(1))
        lane_pos_match = position_pattern.findall(road_file_lines[base_idx+1])
        for s, x, y in lane_pos_match:
            if s == 'StartPos':
                lane_start_pos_x = float(x)
                lane_start_pos_y = float(y)
            if s == 'EndPos':
                lane_end_pos_x = float(x)
                lane_end_pos_y = float(y)
        lane_width_match = lane_width_pattern.findall(road_file_lines[base_idx+3])
        for s, x in lane_width_match:
            if s == 'LaneWidth':
                lane_width = float(x)
        lane_edge_width_match = lane_width_pattern.findall(road_file_lines[base_idx+5])
        for s, x in lane_edge_width_match:
            if s == 'LeftEdgeLineWidth':
                left_edge_width = float(x)
            if s == 'RightEdgeLineWidth':
                right_edge_width = float(x)
        lane_details.append({'lane_id': lane_id, 'scenario': scenario, 'StartPos_x': lane_start_pos_x, 'StartPos_y': lane_start_pos_y,
        'EndPos_x': lane_end_pos_x, 'EndPos_y': lane_end_pos_y, 'LaneWidth': lane_width, 'LeftEdgeLineWidth': left_edge_width,
        'RightEdgeLineWidth': right_edge_width})
    return lane_details


def parse_road_sign_details(road_sign_details, road_file_lines, idx, scenario):
    signs_pattern = re.compile('Number of Signs[\s]*([0-9]+)')
    sign_id_pattern = re.compile('SignId[\s]*([0-9]+)')
    sign_type_pattern = re.compile('SignType[\s]*([0-9]+)')
    sign_position_pattern = re.compile('X[\s]+([\-0-9]+.[0-9]+)[\s]+Y[\s]+([\-0-9]+.[0-9]+)')
    signs_match = signs_pattern.search(road_file_lines[idx])
    if signs_match:
        nr_road_signs = int(signs_match.group(1))
    sign_id = None
    sign_type = None
    xpos = None
    ypos = None
    for i in range(nr_road_signs):
        base_idx = idx+i*3
        sign_id_match = sign_id_pattern.search(road_file_lines[base_idx+1])
        sign_type_match = sign_type_pattern.search(road_file_lines[base_idx+1])
        if sign_id_match:
            sign_id = int(sign_id_match.group(1))
        if sign_type_match:
            sign_type = int(sign_type_match.group(1))
        position_match = sign_position_pattern.search(road_file_lines[base_idx+2])
        if position_match:
            xpos = float(position_match.group(1))
            ypos = float(position_match.group(2))
        road_sign_details.append({'sign_id': sign_id, 'scenario': scenario, 'signType': sign_type, 'sign_xPos': xpos, 'sign_yPos': ypos})
    return road_sign_details


def parse_scenario_information():
    BASE_PATH = 'in/scenario_files/'
    SCENARIO_FILES = {"highway": {"road_file": "highwayLongZWI.net", "scenario_file": "HighwayExp-c.scn"},
                    "rural": {"road_file": "ruralZWI.net", "scenario_file": "RuralExp-c.scn"},
                    "town": {"road_file": "dorpZWI.net", "scenario_file": "TownExp-c.scn"}}

    route_details_rep = []
    route_details_init = []
    path_details = []
    segment_details = []
    lane_details = []
    road_sign_details = []
    for scenario_key in SCENARIO_FILES:

        with open(BASE_PATH + SCENARIO_FILES[scenario_key]["road_file"], 'r') as f:
            road_file_lines = [l.strip() for l in f.readlines()]
        with open(BASE_PATH + SCENARIO_FILES[scenario_key]["scenario_file"], 'r') as f:
            scenario_file_lines = [l.strip() for l in f.readlines()]

        in_rep_function = False
        in_init_function = False
        for idx in range(len(scenario_file_lines)):
            if 'Define Function SetRepeatedRoute()' in scenario_file_lines[idx]:
                in_rep_function = True

            if in_rep_function and ('Part[MainTarget].PathNr' in scenario_file_lines[idx]
                or 'Part[MainTarget].Route' in scenario_file_lines[idx]):
                route_details_rep = parse_route_details(route_details_rep, scenario_file_lines, idx, scenario_key)

            if in_rep_function and "LastResetPath" in scenario_file_lines[idx]:
                in_rep_function = False

            if 'Define Function SetInitRoute()' in scenario_file_lines[idx]:
                in_init_function = True

            if in_init_function and ('Part[MainTarget].PathNr' in scenario_file_lines[idx]
                or 'Part[MainTarget].Route' in scenario_file_lines[idx]):
                route_details_init = parse_route_details(route_details_init, scenario_file_lines, idx, scenario_key)

            if in_init_function and "LastResetPath" in scenario_file_lines[idx]:
                break
        
        route_details = route_details_init + route_details_rep

        for idx in range(len(road_file_lines)):
            if 'Path Nr.' in road_file_lines[idx]:
                path_details = parse_path_details(path_details, road_file_lines, idx, scenario_key)
            if 'Segment Nr.' in road_file_lines[idx]:
                segment_details = parse_segment_details(segment_details, road_file_lines, idx, scenario_key)
            if 'NrDLanes' in road_file_lines[idx]:
                lane_details = parse_lane_details(lane_details, road_file_lines, idx, scenario_key)
            if 'Number of Signs' in road_file_lines[idx] and idx > 10:
                road_sign_details = parse_road_sign_details(road_sign_details, road_file_lines, idx, scenario_key)

    route_df = pd.DataFrame(route_details).drop_duplicates()
    path_df = pd.DataFrame(path_details)
    segment_df = pd.DataFrame(segment_details)
    lane_df = pd.DataFrame(lane_details)

    final_df = segment_df.merge(lane_df, on=['scenario', 'lane_id'], suffixes=['_segment', '_lane'])
    final_df = path_df.merge(final_df, on=['scenario', 'segment_id'])
    final_df = route_df.merge(final_df, on=['scenario', 'path_id'])
    final_df.to_csv('out/scenario_information.csv', index=False)

    road_sign_df = pd.DataFrame(road_sign_details)
    road_sign_df.to_csv('out/road_sign_information.csv', index=False)
