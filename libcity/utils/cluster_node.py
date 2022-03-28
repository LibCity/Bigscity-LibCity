import math
from collections import deque
from tqdm import tqdm

GROUP_SIZE_THRESHOLD = 2  # group size threshold φ
COHERENCE_THRESHOLD = 0.49  # coherence threshold τ
SCALING_FACTOR = 1.1  # scaling factor δ
TURNING_ALPHA = 5  # tuning parameter α
TURNING_BETA = 2  # tuning parameter β
RADIUS = SCALING_FACTOR * \
         ((-math.log(COHERENCE_THRESHOLD)) ** (1 / TURNING_ALPHA))


def range_query(road_gps, point, radius):
    result = []
    for points in road_gps:
        if (road_gps[points][0] > road_gps[point][0] - radius) and (road_gps[points][0] < road_gps[point][0] + radius) \
                and (road_gps[point][1] - radius < road_gps[points][1]) and (
                road_gps[points][1] < road_gps[point][1] + radius):
            result.append(points)
    return result


class Cluster:
    """
    Find road intersections(clusters) from trajectory points through
    coherence expanded algorithm.
    """

    def __init__(self, road_gps, move_direction):
        self.points = road_gps
        self.moving_direction = move_direction
        self.classified = [False for num in range(len(self.points))]

    def coherence_expanding(self):
        clusters = []
        for point in tqdm(self.points, desc='simplify road gps dict'):
            if not self.classified[point]:
                self.classified[point] = True
                cluster = self.expand(point)
                for point in cluster:
                    self.classified[point] = True
                clusters.append(cluster)
        return clusters

    def expand(self, point):
        result = set()
        # save points that has been checked
        searched = set()
        # save point.id
        seeds = deque()
        # save point obejects to be checked

        searched.add(point)
        seeds.append(point)
        result.add(point)

        while (len(seeds)):
            seed = seeds.popleft()

            # find points nearby
            points = range_query(self.points, seed, RADIUS / 100)
            for pt in points:
                coherence = self.__calculate_coherence(seed, pt)
                if coherence >= COHERENCE_THRESHOLD \
                        and (not self.classified[pt]) \
                        and pt not in seeds \
                        and pt not in searched:
                    seeds.append(pt)
                    result.add(pt)
                searched.add(pt)
        return list(result)

    def __calculate_coherence(self, p, q):
        coherence = math.exp(- (self.__distance(p, q) / SCALING_FACTOR) ** TURNING_ALPHA) \
                    * (self.__angle_sin_value(p, q) ** TURNING_BETA)
        return coherence

    def __distance(self, p, q):
        delta_x = self.points[p][0] - self.points[q][0]
        delta_y = self.points[p][1] - self.points[q][1]
        return math.sqrt(delta_x ** 2 + delta_y ** 2)

    def __angle_sin_value(self, p, q):
        x1 = self.moving_direction[p][0] * 1000000
        y1 = self.moving_direction[p][1] * 1000000
        x2 = self.moving_direction[q][0] * 1000000
        y2 = self.moving_direction[q][1] * 1000000

        module_x = math.sqrt(x1 ** 2 + y1 ** 2)
        module_y = math.sqrt(x2 ** 2 + y2 ** 2)
        try:
            angle_cos_value = (x1 * x2 + y1 * y2) / (module_x * module_y)
        except ZeroDivisionError:
            return 0

        return math.sqrt(abs(1 - angle_cos_value ** 2))
