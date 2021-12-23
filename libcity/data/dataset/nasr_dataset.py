

def distance_to_bin(distance_x):
    """
    discrete distance between road
    The bin size is 500 m.
    For distances over 10 km, all are mapped into a same bucket.

    Args:
        distance_x: the unit is meter.

    Returns:
        distance_bin
    """
    if distance_x >= 10000:
        return 20
    else:
        return distance_x // 500
