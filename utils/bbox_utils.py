def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return x2 - x1


def measure_distance(box1, box2):
    return ((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5


def measure_xy_distance(point1, point2):
    return point1[0] - point2[0], point1[1] - point2[1]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
