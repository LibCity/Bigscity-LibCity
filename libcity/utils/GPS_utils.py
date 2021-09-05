import math

R_EARTH = 6371000  # meter


def angle2radian(angle):
    """
    convert from an angle to a radian
    :param angle: (float)
    :return: radian (float)
    """
    return math.radians(angle)


def radian2angle(radian):
    return math.degrees(radian)


def spherical_law_of_cosines(phi1, lambda1, phi2, lambda2):
    """
    calculate great circle distance with spherical law of cosines
    phi/lambda for latitude/longitude in radians
    :param phi1: point one's latitude in radians
    :param lambda1: point one's longitude in radians
    :param phi2: point two's latitude in radians
    :param lambda2: point two's longitude in radians
    :return:
    """
    d_lambda = lambda2 - lambda1
    return math.acos(math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(d_lambda))


def haversine(phi1, lambda1, phi2, lambda2):
    """
    calculate angular great circle distance with haversine formula
    see parameters in spherical_law_of_cosines
    """
    d_phi = phi2 - phi1
    d_lambda = lambda2 - lambda1
    a = math.pow(math.sin(d_phi / 2), 2) + \
        math.cos(phi1) * math.cos(phi2) * math.pow(math.sin(d_lambda / 2), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return c


def equirectangular_approximation(phi1, lambda1, phi2, lambda2):
    """
    calculate angular great circle distance with Pythagoras’ theorem performed on an equirectangular projection
    see parameters in spherical_law_of_cosines
    """
    x = (lambda2 - lambda1) * math.cos((phi1 + phi2) / 2)
    y = phi2 - phi1
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))


def dist(phi1, lambda1, phi2, lambda2, r=R_EARTH, method='hav'):
    """
    calculate great circle distance with given latitude and longitude,
    :param phi1: point one's latitude in angle
    :param lambda1: point one's longitude in angle
    :param phi2: point two's latitude in angle
    :param lambda2: point two's longitude in angle
    :param r: earth radius(m)
    :param method:  'hav' means haversine,
                    'LoC' means Spherical Law of Cosines,
                    'approx' means Pythagoras’ theorem performed on an equirectangular projection
    :return: distance (m)
    """
    return angular_dist(phi1, lambda1, phi2, lambda2, method) * r


def angular_dist(phi1, lambda1, phi2, lambda2, method='hav'):
    """
    calculate angular great circle distance with given latitude and longitude
    :return: angle
    """
    if method.lower() == 'hav':
        return haversine(phi1, lambda1, phi2, lambda2)
    elif method.lower() == 'loc':
        return spherical_law_of_cosines(phi1, lambda1, phi2, lambda2)
    elif method.lower() == 'approx':
        return equirectangular_approximation(phi1, lambda1, phi2, lambda2)
    else:
        assert False


def destination(phi1, lambda1, brng, distance, r=R_EARTH):
    """

    :param phi1:
    :param lambda1:
    :param brng:
    :param distance:
    :return:
    """
    delta = distance / r
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(brng))
    lambda2 = lambda1 + math.atan2(
        math.sin(brng) * math.sin(delta) * math.cos(phi1), math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    )
    return phi2, lambda2


def init_bearing(phi1, lambda1, phi2, lambda2):
    """
    initial bearing of a great circle route
    :return: 0~360
    """
    y = math.sin(lambda2 - lambda1) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(lambda2 - lambda1)
    theta = math.atan2(y, x)
    brng = (theta * 180 / math.pi + 360) % 360
    return brng