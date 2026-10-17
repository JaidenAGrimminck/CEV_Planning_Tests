import bisect
import math
import random
import numpy as np
from dataclasses import dataclass

import pygame
SCREEN_WIDTH = 2000
SCREEN_HEIGHT = 1000
SCREEN_SIZE = (2000, 1000)
SCALE = 0.7
FPS = 60
TARGET = (1950, 700)
TARGET_RAD = (70)
TARGET_SPEED = 5
# Car geometry (pixels)
CAR_LENGTH = 20
CAR_WIDTH = 10
WHEELBASE = 3  # distance between front and rear axles
WHEEL_LENGTH = 5
WHEEL_WIDTH = 2

DISTANCE_TIME_PROP = 1
delta_B_N = 7 
delta_S = 7 
TIME_PROP = 0.5
# Car dynamics
MAX_ACCELERATION = 120  # px/s^2, limit on set_acceleration()
MAX_SPEED = 300  # px/s
MAX_REVERSE_SPEED = 100
MAX_STEER_RATE = math.radians(20)  # rad/s, limit on set_steering_rate()
MAX_STEER = math.radians(35)  # rad, limit on front wheel steering angle
MAX_LATERAL_ACCELERATION = 200
SDF_CAP = 10

# Obstacles: [x, y] centers and radii
GLOBAL_OBSTACLE_POS = [[574, 109], [1592, 867], [1425, 537], [1579, 85], [438, 146], [1885, 883], [894, 870], [1414, 70], [569, 816], [433, 522], [364, 841], [1515, 785], [1123, 921], [968, 218], [1412, 199], [1614, 521], [1397, 843], [88, 873], [1031, 834], [1128, 503], [1876, 520], [1100, 219], [892, 113], [1073, 81], [608, 518], [70, 91], [1925, 96], [122, 525], [460, 866], [971, 452], [884, 547], [1555, 213], [634, 937], [1512, 450], [361, 60], [129, 202], [1019, 946], [129, 433], [1060, 567], [1343, 454], [1647, 211], [1494, 894], [495, 45], [343, 195], [1665, 786], [449, 779], [1947, 430], [951, 780], [1901, 215], [846, 216], [970, 568], [375, 946], [353, 442], [119, 772], [1412, 436], [645, 39], [536, 235], [1083, 422], [1138, 787], [867, 431], [151, 956], [159, 824], [41, 473], [53, 247], [654, 855], [1854, 434], [1931, 761], [498, 944], [1861, 759], [1605, 783], [1486, 958], [1520, 569], [1158, 159], [663, 425], [628, 222], [1366, 958], [1006, 162], [439, 241], [1854, 38], [1650, 961], [333, 517], [866, 962], [1557, 934], [1850, 210], [382, 249], [660, 173], [42, 943], [439, 46], [1972, 233], [604, 413], [578, 894], [1859, 247], [647, 794], [1662, 907], [839, 799], [1116, 582], [1014, 747], [550, 932], [511, 414], [576, 24], [56, 520], [526, 460], [1340, 109], [351, 577], [172, 77], [1173, 108], [1958, 564], [1498, 165], [670, 94], [1959, 811], [1427, 934], [58, 798], [1975, 151], [332, 751], [965, 909], [149, 36], [868, 751], [77, 580], [1834, 132], [1021, 527], [1486, 64], [1616, 940], [457, 428], [1325, 68], [594, 747], [383, 770], [1361, 752], [1953, 956], [1969, 501], [1668, 25], [176, 481], [1616, 424], [1329, 910], [59, 410], [46, 763], [832, 459], [455, 971], [1030, 239], [47, 184], [661, 134], [329, 125], [1671, 169], [1134, 30], [904, 244], [1488, 123], [1162, 428], [976, 53], [1350, 28], [955, 945], [1860, 593], [1837, 964], [167, 775], [409, 417], [1085, 762], [498, 581], [1339, 157], [1964, 23], [1338, 573], [1338, 255], [994, 120], [154, 117]]

GLOBAL_OBSTACLE_RAD = [64, 45, 63, 73, 57, 71, 61, 46, 54, 62, 46, 47, 56, 45, 63, 59, 74, 56, 60, 56, 46, 46, 69, 63, 70, 52, 53, 48, 49, 47, 52, 50, 49, 48, 50, 54, 41, 34, 27, 32, 29, 28, 29, 27, 25, 34, 28, 28, 31, 35, 27, 35, 40, 29, 26, 26, 33, 25, 36, 27, 27, 25, 30, 26, 31, 31, 32, 30, 26, 28, 28, 32, 31, 27, 40, 29, 16, 15, 17, 15, 16, 22, 23, 15, 17, 12, 14, 20, 15, 17, 13, 14, 19, 18, 14, 18, 14, 13, 15, 14, 13, 20, 25, 13, 17, 17, 24, 13, 12, 14, 13, 19, 12, 21, 14, 16, 21, 17, 21, 22, 17, 15, 23, 13, 16, 21, 15, 21, 16, 15, 12, 19, 17, 15, 13, 14, 17, 12, 14, 14, 17, 14, 13, 17, 13, 21, 12, 12, 19, 14, 14, 12, 18, 16, 18, 16, 12, 13, 17, 16, 19]

LOCAL_OBSTACLE_POS = [[784, 116], [889, 692], [1258, 643], [277, 763], [697, 676], [941, 287], [1200, 663], [1649, 714], [1715, 160], [1780, 697], [1781, 628], [1800, 76], [786, 772], [804, 47], [1002, 614], [611, 290], [1695, 939], [1541, 307], [1194, 881], [1214, 192], [1247, 378], [106, 623], [1194, 584], [1178, 711], [180, 310], [787, 475], [1291, 41], [239, 627], [1886, 380], [343, 647], [1769, 740], [1719, 275], [219, 174], [628, 718], [1420, 304], [725, 403], [1269, 295], [1288, 129], [283, 917], [1206, 292], [856, 286], [487, 283], [1826, 279], [1195, 39], [1717, 674], [1773, 836], [1702, 569], [806, 824], [1709, 828], [1555, 366], [288, 468], [1779, 379], [1451, 624], [1973, 722], [107, 698], [1227, 725], [198, 385], [298, 574], [1016, 387], [214, 563], [1560, 715], [1296, 512], [674, 620], [630, 360], [47, 631], [1659, 280], [305, 359], [727, 66], [216, 494], [1077, 621], [689, 359], [1905, 698], [1793, 952], [1591, 630], [124, 286], [798, 632], [821, 716], [1187, 361], [737, 279], [1778, 539], [228, 292], [954, 710], [1893, 302], [1211, 516], [1707, 79], [1705, 421], [722, 825], [1695, 616], [29, 356], [1950, 633], [288, 146], [1965, 296], [559, 371], [1028, 710], [1769, 223], [230, 706], [426, 723], [296, 708], [210, 938], [701, 550], [1723, 761], [902, 625], [429, 296], [730, 329], [1826, 719], [572, 648], [1311, 374], [273, 257], [1392, 628], [809, 406], [915, 375], [1274, 183], [768, 963], [1298, 871], [1960, 367], [192, 665], [210, 765], [698, 224], [270, 856], [717, 908], [1800, 180], [1808, 464], [297, 204], [1223, 448], [418, 368], [1317, 714], [1479, 282], [776, 880], [1408, 360], [696, 469], [1731, 349], [1292, 821], [694, 136], [447, 614], [675, 303], [1413, 696], [363, 377], [952, 610], [1803, 342], [1279, 597], [793, 351], [1719, 218], [855, 632], [1704, 520], [1846, 621], [1497, 378], [1019, 273], [208, 55], [1217, 974], [1089, 688], [25, 282], [1138, 306], [1111, 726], [352, 290], [726, 642], [167, 619], [1528, 634], [795, 547], [1132, 643], [1285, 459], [1214, 88], [1488, 708], [195, 113], [629, 611], [1327, 636], [1291, 934], [549, 710], [173, 717], [206, 863], [199, 235], [1068, 310], [491, 363], [729, 179], [1902, 638], [1833, 369], [753, 703], [1623, 372], [236, 978], [1362, 317], [1588, 282], [728, 748], [1723, 891], [296, 972], [106, 386], [1214, 810], [273, 409], [1767, 491], [1281, 751], [48, 714], [486, 719], [1098, 372], [283, 86], [691, 973], [211, 436], [505, 624], [1697, 724], [785, 173], [1032, 648], [1786, 786], [1780, 298], [805, 935], [782, 302], [358, 710], [792, 255], [297, 26], [546, 313], [1638, 608], [68, 304], [863, 376], [158, 367], [730, 949], [1726, 32], [1299, 243], [390, 628], [1232, 22], [1273, 695]]

LOCAL_OBSTACLE_RAD = [17, 12, 10, 17, 10, 17, 9, 13, 17, 10, 16, 10, 11, 16, 8, 9, 15, 19, 16, 13, 18, 15, 13, 9, 7, 13, 10, 18, 12, 11, 8, 16, 19, 9, 16, 17, 11, 19, 9, 18, 18, 15, 9, 9, 8, 12, 8, 18, 19, 9, 15, 12, 9, 11, 19, 16, 8, 14, 16, 12, 16, 14, 8, 12, 16, 10, 20, 13, 10, 16, 14, 13, 17, 12, 19, 12, 16, 9, 12, 11, 10, 14, 9, 8, 11, 16, 13, 7, 10, 10, 17, 9, 19, 14, 9, 17, 19, 17, 13, 19, 11, 10, 13, 10, 15, 13, 19, 11, 16, 20, 17, 12, 9, 8, 14, 11, 8, 17, 13, 12, 18, 16, 13, 12, 14, 15, 14, 9, 11, 18, 10, 10, 11, 13, 14, 15, 11, 8, 8, 16, 10, 12, 8, 16, 10, 19, 15, 16, 12, 7, 7, 18, 10, 14, 8, 15, 15, 9, 15, 16, 15, 12, 12, 8, 12, 13, 16, 14, 10, 7, 8, 10, 11, 13, 9, 9, 16, 8, 9, 10, 9, 12, 18, 10, 14, 13, 8, 11, 14, 10, 20, 10, 12, 8, 19, 7, 13, 12, 9, 14, 13, 10, 10, 10, 15, 11, 9, 14, 10, 11, 7, 12, 11, 12, 7, 8]


@dataclass(frozen=True)
class CarState:
    x: float  # px, rear axle center
    y: float  # px
    velocity: float  # px/s, negative = reversing
    heading: float  # rad, 0 = +x, positive = clockwise on screen


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


OBSTACLES = [Obstacle(pos[0], pos[1], rad) for pos, rad in zip(GLOBAL_OBSTACLE_POS + LOCAL_OBSTACLE_POS, GLOBAL_OBSTACLE_RAD + LOCAL_OBSTACLE_RAD)]
GLOBAL_OBSTACLES = [Obstacle(pos[0], pos[1], rad) for pos, rad in zip(GLOBAL_OBSTACLE_POS, GLOBAL_OBSTACLE_RAD)]
Sorted_x_obstacles = sorted(OBSTACLES, key=lambda o: o.x)
Sorted_y_obstacles = sorted(OBSTACLES, key=lambda o: o.y)
max_radius = max(o.radius for o in OBSTACLES)


class Car:
    """Kinematic Ackermann model driven by front wheel steering angle. The pose is defined at the rear axle center.

    Control it by setting rates; update() integrates them each frame:
        car.set_acceleration(px_per_s2)    # clamped to +/-MAX_ACCELERATION
        car.set_steering_rate(rad_per_s)  # d(delta)/dt, clamped to +/-MAX_STEER_RATE
        car.get_state() -> CarState
    """

    def __init__(self, x, y, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = 0.0
        self.steering = 0.0  # rad, front wheel steering angle delta
        self.acceleration = 0.0  # px/s^2, current speed rate of change
        self.steering_rate = 0.0  # rad/s, current steering angle rate of change

    def get_state(self):
        return CarState(self.x, self.y, self.velocity, self.heading)

    def set_acceleration(self, acceleration):
        """Rate of change of speed in px/s^2. Returns the clamped value actually applied."""
        self.acceleration = clamp(acceleration, -MAX_ACCELERATION, MAX_ACCELERATION)
        return self.acceleration

    def set_steering_rate(self, steering_rate):
        """Rate of change of steering angle in rad/s (positive = right). Returns the clamped value actually applied."""
        self.steering_rate = clamp(steering_rate, -MAX_STEER_RATE, MAX_STEER_RATE)
        return self.steering_rate

    def update(self, dt):
        self.velocity = clamp(self.velocity + self.acceleration * dt, -MAX_SPEED, MAX_SPEED)
        self.steering = clamp(self.steering + self.steering_rate * dt, -MAX_STEER, MAX_STEER)

        self.x += self.velocity * math.cos(self.heading) * dt
        self.y += self.velocity * math.sin(self.heading) * dt
        self.heading += self.velocity * math.tan(self.steering) / WHEELBASE * dt
        self.heading %= 2 * math.pi

    def draw(self, surface):
        cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)

        def to_world(forward, left):
            # Body frame (forward along heading, left = toward -y side) -> screen coords
            return (
                self.x + forward * cos_h + left * sin_h,
                self.y + forward * sin_h - left * cos_h,
            )

        # Body: centered between the axles
        overhang = (CAR_LENGTH - WHEELBASE) / 2
        body_center = to_world(WHEELBASE / 2, 0)
        pygame.draw.polygon(surface, "red", rotated_rect(body_center, CAR_LENGTH, CAR_WIDTH, self.heading))

        # Wheels: rear follow the body, front use Ackermann geometry so each wheel
        # points around the same turning center (inner wheel turns more than outer)
        half_track = CAR_WIDTH / 2
        tan_steer = math.tan(self.steering)
        for left in (half_track, -half_track):
            wheel_steer = math.atan(WHEELBASE * tan_steer / (WHEELBASE + left * tan_steer))
            for forward, wheel_angle in ((0, self.heading), (WHEELBASE, self.heading + wheel_steer)):
                wheel = rotated_rect(to_world(forward, left), WHEEL_LENGTH, WHEEL_WIDTH, wheel_angle)
                pygame.draw.polygon(surface, "white", wheel)

        # Nose marker so heading is obvious
        pygame.draw.circle(surface, "yellow", to_world(WHEELBASE + overhang, 0), 2)


def clamp(value, low, high):
    return max(low, min(high, value))


def rotated_rect(center, length, width, angle):
    """Corners of a length x width rectangle centered at `center`, rotated by `angle`."""
    cx, cy = center
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = []
    for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        fx, fy = dx * length / 2, dy * width / 2
        corners.append((cx + fx * cos_a - fy * sin_a, cy + fx * sin_a + fy * cos_a))
    return corners


def draw_obstacles(surface):
    for pos, rad in zip(GLOBAL_OBSTACLE_POS, GLOBAL_OBSTACLE_RAD):
        pygame.draw.circle(surface, "blue", pos, rad)
    for pos, rad in zip(LOCAL_OBSTACLE_POS, LOCAL_OBSTACLE_RAD):
        pygame.draw.circle(surface, "green", pos, rad)


def load_font():
    # pygame 2.6.1's font module is broken on Python 3.14; fall back to the window title
    try:
        return pygame.font.SysFont(None, 28)
    except (NotImplementedError, ImportError):
        return None


def draw_hud(surface, font, car):
    text = f"x: {car.x:7.1f}   y: {car.y:7.1f}   heading: {math.degrees(car.heading):6.1f} deg   speed: {car.velocity:6.1f} px/s   steering: {math.degrees(car.steering):6.1f} deg"
    if font:
        surface.blit(font.render(text, True, "white"), (10, 10))
    else:
        pygame.display.set_caption(text)


class SDFState:
    def __init__(self, obstacles):
        self.obstacles = list(obstacles)
        self.matrix = np.full((SCREEN_WIDTH, SCREEN_HEIGHT), SDF_CAP, dtype=np.float32)
        self.has_changed = False
        self.change_coords = []
        self.computeSDF()

    def computeSDF(self, obstacles=None):
        sdf = self.matrix
        for o in self.obstacles if obstacles is None else obstacles:
            reach = o.radius + SDF_CAP
            x_lo, x_hi = max(0, math.floor(o.x - reach)), min(SCREEN_WIDTH, math.ceil(o.x + reach) + 1)
            y_lo, y_hi = max(0, math.floor(o.y - reach)), min(SCREEN_HEIGHT, math.ceil(o.y + reach) + 1)
            if x_lo >= x_hi or y_lo >= y_hi:
                continue
            xs = np.arange(x_lo, x_hi)[:, None]
            ys = np.arange(y_lo, y_hi)[None, :]
            dist = (np.hypot(xs - o.x, ys - o.y) - o.radius).astype(np.float32)
            np.minimum(sdf[x_lo:x_hi, y_lo:y_hi], dist, out=sdf[x_lo:x_hi, y_lo:y_hi])
        return self

    def addObstacles(self, obstacles):
        self.obstacles.extend(obstacles)
        self.computeSDF(obstacles)
        self.change_coords.extend((o.x, o.y) for o in obstacles)
        self.has_changed = True
        return self


class state:
    
    def __init__(self, x1, x2, a, s, cost = 0, steer = 0):
        self.x1 = x1
        self.x2 = x2 
        self.a = a 
        self.s = s
        self.x1bar = self.x1 +self.s * math.cos(self.a)*DISTANCE_TIME_PROP
        self.x2bar = self.x2 + self.s * math.sin(self.a)*DISTANCE_TIME_PROP 
        self.cost = cost #we need to define cost
        self.steer = steer
        self.children =  []
        self.parent = None

    def distance(self, s2):
        return calcReedsSheppDist(self, s2)
    
    def removeUntilActive(self, tree):
        tree.inactive.discard(self)
        if(self.parent!=None and (self.parent) in tree.inactive):
            self.parent.removeUntilActive(tree)
    
    
class action:
    def __init__(self, da, ds):
        self.da = da 
        self.ds = ds 

class edge:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2


class SSTTree:
    def __init__(self, s1):
        self.active = {s1}

        self.inactive = set()
    
        self.E = np.array([])
        
    def activate(self, vertex):
        self.active.discard(vertex)
        self.inactive.add(vertex)
    def unActivate(self,vertex):
        self.inactive.discard(vertex)
        self.active.add(vertex)
        
class proximityHashMap:
    def __init__(self, delta):
        self.rad = delta
        self.map = {}
    def add(self, s):
        x_coord = math.floor(s.x1/self.rad)
        y_coord = math.floor(s.x2/self.rad)
        if x_coord not in self.map:
            self.map[x_coord] = {}
        if y_coord in self.map[x_coord]:
            self.map[x_coord][y_coord].append(s)

        else:
            self.map[x_coord][y_coord] = [s]
            
    def get(self, x_coord, y_coord):
        if x_coord in self.map and y_coord in self.map[x_coord]:
            return self.map[x_coord][y_coord]
        return []
            
    def get_nbhd(self, s, delta):
        x_lo = math.floor((s.x1 - delta)/self.rad)
        x_hi = math.floor((s.x1 + delta)/self.rad)
        y_lo = math.floor((s.x2 - delta)/self.rad)
        y_hi = math.floor((s.x2 + delta)/self.rad)
        result = []
        for i in range(x_lo, x_hi + 1):
            for j in range(y_lo, y_hi + 1):
                for other in self.get(i, j):
                    if math.hypot(other.x1 - s.x1, other.x2 - s.x2) <= delta:
                        result.append(other)
                
        return result 




    
    
            
        

def getRandomState():
    x_1_rand = random.uniform(0, SCREEN_WIDTH)
    x_2_rand = random.uniform(0, SCREEN_HEIGHT)
    a_rand = random.uniform(0, 2 * math.pi)
    speed_rand = random.uniform(0, 10) #ask claude how we should define this 

    return state(x_1_rand, x_2_rand, a_rand, speed_rand)

def maxSteerAtSpeed(speed):
    if speed == 0:
        return MAX_STEER
    return min(MAX_STEER, math.atan(MAX_LATERAL_ACCELERATION * WHEELBASE / speed ** 2))


def getRandomAction(x_selected, time):
    steer_needed = max(0, abs(x_selected.steer) - MAX_STEER_RATE * time)
    speed_allowed = math.inf if steer_needed == 0 else math.sqrt(MAX_LATERAL_ACCELERATION * WHEELBASE / math.tan(steer_needed))
    accel_lo, accel_hi = -MAX_ACCELERATION, MAX_ACCELERATION
    if time > 0:
        accel_lo = max(accel_lo, (max(-MAX_REVERSE_SPEED, -speed_allowed) - x_selected.s) / time)
        accel_hi = min(accel_hi, (min(MAX_SPEED, speed_allowed) - x_selected.s) / time)
    acceleration_rand = random.uniform(accel_lo, accel_hi) #ask claude how we should define this
    steer_limit = maxSteerAtSpeed(max(abs(x_selected.s), abs(x_selected.s + acceleration_rand * time)))
    steer_rand = random.uniform(max(-steer_limit, x_selected.steer - MAX_STEER_RATE * time), min(steer_limit, x_selected.steer + MAX_STEER_RATE * time))

    return action(steer_rand, acceleration_rand)

def monteCarloProp(x_selected, sdf_state):
    time = random.uniform(0, TIME_PROP)
    action = getRandomAction(x_selected, time)
    curvature = math.tan(action.da) / WHEELBASE
    for i in range(1, 6):
        t = time * i / 5
        dist = x_selected.s * t + action.ds * t**2 / 2
        if abs(action.da) < 1e-9:
            x1_final = x_selected.x1 + dist * math.cos(x_selected.a)
            x2_final = x_selected.x2 + dist * math.sin(x_selected.a)
            a_final = x_selected.a
        else:
            a_final = x_selected.a + curvature * dist
            x1_final = x_selected.x1 + (math.sin(a_final) - math.sin(x_selected.a)) / curvature
            x2_final = x_selected.x2 - (math.cos(a_final) - math.cos(x_selected.a)) / curvature
        px, py = round(x1_final), round(x2_final)
        if not (0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT) or sdf_state.matrix[px, py] <= 0:
            return state(-1, -1, x_selected.a, x_selected.s)
    #check for possible ersrors?
    return state(x1_final, x2_final, a_final, x_selected.s + action.ds * time, steer=action.da, cost=x_selected.cost+time)


def near(nodeList, hash_map, s1):
    candidates = hash_map.get_nbhd(s1, MAX_SPEED * delta_B_N)
    if len(candidates) == 0:
        candidates = nodeList
    x_near = []
    min = float('inf')
    min_node = candidates[0]
    for i in range(len(candidates)):
        dist = candidates[i].distance(s1)
        if dist <= delta_B_N:
            x_near.append(candidates[i])
        if dist<=min:
            min = dist
            min_node = candidates[i]

    return x_near, min_node
        

def bestFirstSelectionSST(t, hash_map):
    x_rand = getRandomState()
    x_near, min_node = near(t, hash_map, x_rand)
    
    if len(x_near) == 0:
        return min_node
    else:
        min_cost = float('inf')
        min_cost_index = 0
        for i in range(len(x_near)):
            if x_near[i].cost < min_cost:
                min_cost_index = i
                min_cost = x_near[i].cost
        return x_near[min_cost_index]




def calcReedsSheppDist(s0, s1):
    pi2 = 2 * math.pi
    eps = 1e-9
 
    def mod2pi(x):
        v = math.fmod(x, pi2)
        if v < -math.pi:
            v += pi2
        elif v > math.pi:
            v -= pi2
        return v
 
    def polar(x, y):
        return math.hypot(x, y), math.atan2(y, x)
 
    def tau_omega(u, v, xi, eta, phi):
        delta = mod2pi(u - v)
        a = math.sin(u) - math.sin(delta)
        b = math.cos(u) - math.cos(delta) - 1
        t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
        t2 = 2 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3
        tau = mod2pi(t1 + math.pi) if t2 < 0 else mod2pi(t1)
        return tau, mod2pi(tau - u + v - phi)
 
    def lsl(x, y, phi):
        u, t = polar(x - math.sin(phi), y - 1 + math.cos(phi))
        if t >= -eps:
            v = mod2pi(phi - t)
            if v >= -eps:
                return [t, u, v]
        return None
 
    def lsr(x, y, phi):
        u1, t1 = polar(x + math.sin(phi), y - 1 - math.cos(phi))
        u1 = u1 * u1
        if u1 >= 4:
            u = math.sqrt(u1 - 4)
            t = mod2pi(t1 + math.atan2(2, u))
            v = mod2pi(t - phi)
            if t >= -eps and v >= -eps:
                return [t, u, v]
        return None
 
    def lrl(x, y, phi):
        xi = x - math.sin(phi)
        eta = y - 1 + math.cos(phi)
        u1, theta = polar(xi, eta)
        if u1 <= 4:
            u = -2 * math.asin(0.25 * u1)
            t = mod2pi(theta + 0.5 * u + math.pi)
            v = mod2pi(phi - t + u)
            if t >= -eps and u <= eps:
                return [t, u, v]
        return None
 
    def lrlr_a(x, y, phi):
        xi = x + math.sin(phi)
        eta = y - 1 - math.cos(phi)
        rho = 0.25 * (2 + math.hypot(xi, eta))
        if rho <= 1:
            u = math.acos(rho)
            t, v = tau_omega(u, -u, xi, eta, phi)
            if t >= -eps and v <= eps:
                return [t, u, -u, v]
        return None
 
    def lrlr_b(x, y, phi):
        xi = x + math.sin(phi)
        eta = y - 1 - math.cos(phi)
        rho = (20 - xi * xi - eta * eta) / 16
        if 0 <= rho <= 1:
            u = -math.acos(rho)
            if u >= -math.pi / 2:
                t, v = tau_omega(u, u, xi, eta, phi)
                if t >= -eps and v >= -eps:
                    return [t, u, u, v]
        return None
 
    def lrsl(x, y, phi):
        xi = x - math.sin(phi)
        eta = y - 1 + math.cos(phi)
        rho, theta = polar(xi, eta)
        if rho >= 2:
            r = math.sqrt(rho * rho - 4)
            u = 2 - r
            t = mod2pi(theta + math.atan2(r, -2))
            v = mod2pi(phi - math.pi / 2 - t)
            if t >= -eps and u <= eps and v <= eps:
                return [t, -math.pi / 2, u, v]
        return None
 
    def lrsr(x, y, phi):
        xi = x + math.sin(phi)
        eta = y - 1 - math.cos(phi)
        rho, theta = polar(-eta, xi)
        if rho >= 2:
            t = theta
            u = 2 - rho
            v = mod2pi(t + math.pi / 2 - phi)
            if t >= -eps and u <= eps and v <= eps:
                return [t, -math.pi / 2, u, v]
        return None
 
    def lrslr(x, y, phi):
        xi = x + math.sin(phi)
        eta = y - 1 - math.cos(phi)
        rho, theta = polar(xi, eta)
        if rho >= 2:
            u = 4 - math.sqrt(rho * rho - 4)
            if u <= eps:
                t = mod2pi(math.atan2((4 - u) * xi - 2 * eta, -2 * xi + (u - 4) * eta))
                v = mod2pi(t - phi)
                if t >= -eps and v >= -eps:
                    return [t, -math.pi / 2, u, -math.pi / 2, v]
        return None
 
    def variants(fn, x, y, phi, backwards=False):
        out = []
        for sx, sy, sp, flip in ((1, 1, 1, False), (-1, 1, -1, True), (1, -1, -1, False), (-1, -1, 1, True)):
            seg = fn(sx * x, sy * y, sp * phi)
            if seg is None:
                continue
            if backwards:
                seg = seg[::-1]
            if flip:
                seg = [-s for s in seg]
            out.append(seg)
        return out
 
    def run_time(length, va, vb, vmax):
        a = MAX_ACCELERATION
        va = min(va, vmax)
        vb = min(vb, vmax)
        if abs(vb * vb - va * va) > 2 * a * length:
            return float('inf')
        v_peak = math.sqrt(a * length + (va * va + vb * vb) / 2)
        if v_peak <= vmax:
            return (2 * v_peak - va - vb) / a
        accel_dist = (vmax * vmax - va * va) / (2 * a)
        decel_dist = (vmax * vmax - vb * vb) / (2 * a)
        return (2 * vmax - va - vb) / a + (length - accel_dist - decel_dist) / vmax
 
    r = WHEELBASE / math.tan(MAX_STEER)
    dx, dy = s1.x1 - s0.x1, s1.x2 - s0.x2
    c, s = math.cos(s0.a), math.sin(s0.a)
    x = (c * dx + s * dy) / r
    y = (-s * dx + c * dy) / r
    phi = mod2pi(s1.a - s0.a)
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
 
    candidates = []
    for fn in (lsl, lsr, lrlr_a, lrlr_b, lrslr):
        candidates += variants(fn, x, y, phi)
    for fn in (lrl, lrsl, lrsr):
        candidates += variants(fn, x, y, phi)
        candidates += variants(fn, xb, yb, phi, backwards=True)
    v0, v1 = s0.s, s1.s
 
    def path_time(segments):
        runs = []
        for seg in segments:
            length = abs(seg) * r
            if length < eps:
                continue
            direction = 1 if seg > 0 else -1
            if runs and runs[-1][0] == direction:
                runs[-1][1] += length
            else:
                runs.append([direction, length])
 
        if not runs:
            return abs(v1 - v0) / MAX_ACCELERATION
 
        total = 0.0
        last = len(runs) - 1
        for i, (direction, length) in enumerate(runs):
            vmax = MAX_SPEED if direction > 0 else MAX_REVERSE_SPEED
            va = 0.0
            vb = 0.0
            if i == 0:
                if v0 * direction >= 0:
                    va = abs(v0)
                else:
                    total += abs(v0) / MAX_ACCELERATION
            if i == last:
                if v1 * direction >= 0:
                    vb = abs(v1)
                else:
                    total += abs(v1) / MAX_ACCELERATION
            total += run_time(length, va, vb, vmax)
        return total
 
    return min(path_time(seg) for seg in candidates)
class Vertex:
    def __init__(self, x1, x2, parent=None, child=None, cost=0):
        self.x1 = x1
        self.x2 = x2
        self.parent = parent
        self.child = [] if child is None else child
        self.cost = cost


class RRTGraph:
    def __init__(self):
        self.vertices = [Vertex(250, 100)]
        self.solved_vertex = None

    def getClosestVertex(self, x1, x2):
        closest = None
        min_dist = float('inf')
        for v in self.vertices:
            dist = math.hypot(v.x1 - x1, v.x2 - x2)
            if dist < min_dist:
                min_dist = dist
                closest = v
        return closest

    def add(self, x1, x2):
        closest = self.getClosestVertex(x1, x2)
        if not getValidityOfPath(closest.x1, closest.x2, x1, x2):
            return False
        new_vertex = Vertex(x1, x2, parent=closest)
        closest.child.append(new_vertex)
        self.vertices.append(new_vertex)
        if math.hypot(x1 - TARGET[0], x2 - TARGET[1]) <= TARGET_RAD:
            self.solved_vertex = new_vertex
        return True


def getValidityOfPath(s1x1, s1x2, s2x1, s2x2):
    min_x = min(s1x1, s2x1) - max_radius
    max_x = max(s1x1, s2x1) + max_radius
    min_y = min(s1x2, s2x2) - max_radius
    max_y = max(s1x2, s2x2) + max_radius

    i = bisect.bisect_left(Sorted_x_obstacles, min_x, key=lambda o: o.x)
    j = bisect.bisect_right(Sorted_x_obstacles, max_x, key=lambda o: o.x) - 1

    dx = s2x1 - s1x1
    dy = s2x2 - s1x2
    length_sq = dx * dx + dy * dy
    for o in Sorted_x_obstacles[i:j + 1]:
        if o.y < min_y or o.y > max_y:
            continue
        t = 0 if length_sq == 0 else clamp(((o.x - s1x1) * dx + (o.y - s1x2) * dy) / length_sq, 0, 1)
        if math.hypot(s1x1 + t * dx - o.x, s1x2 + t * dy - o.y) <= o.radius+7*SCALE:
            return False
    return True


def main():
    rrt_tree = RRTGraph()

    while rrt_tree.solved_vertex is None:
        rrt_tree.add(random.uniform(0, SCREEN_WIDTH), random.uniform(0, SCREEN_HEIGHT))

    Solution_RRT_Path = []
    node = rrt_tree.solved_vertex
    while node is not None:
        Solution_RRT_Path.append(node)
        node = node.parent

    pygame.init()
    window = pygame.display.set_mode((int(SCREEN_WIDTH * SCALE), int(SCREEN_HEIGHT * SCALE)))
    screen = pygame.Surface(SCREEN_SIZE)
    clock = pygame.time.Clock()
    font = load_font()
    car = Car(250, 100)

    dt = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        throttle = keys[pygame.K_w] - keys[pygame.K_s]
        steer = keys[pygame.K_d] - keys[pygame.K_a]
        car.set_acceleration(throttle * MAX_ACCELERATION)
        car.set_steering_rate(steer * MAX_STEER_RATE)
        car.update(dt)

        screen.fill("black")
        draw_obstacles(screen)
        car.draw(screen)
        draw_hud(screen, font, car)
        for v in rrt_tree.vertices:
            if v.parent is not None:
                pygame.draw.line(screen, "gray", (v.parent.x1, v.parent.x2), (v.x1, v.x2), 1)
            pygame.draw.circle(screen, "gray", (v.x1, v.x2), 3)
        pygame.draw.lines(screen, "PURPLE", False, [(v.x1, v.x2) for v in Solution_RRT_Path], 2)
        for v in Solution_RRT_Path:
            pygame.draw.circle(screen, "PURPLE", (v.x1, v.x2), 5)

        pygame.draw.circle(screen, "PURPLE", TARGET, TARGET_RAD)
        window.blit(pygame.transform.smoothscale(screen, window.get_size()), (0, 0))
        pygame.display.flip()

        dt = clock.tick(FPS) / 1000
    pygame.quit()


if __name__ == "__main__":
    main()
