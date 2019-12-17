#!/usr/bin/python3

from __future__ import print_function

from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks

from autobahn.wamp.serializer import MsgPackSerializer
from autobahn.wamp.types import ComponentConfig
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner

import argparse
import random
import math
import sys

import base64
import numpy as np

import helper

import copy

import fuzzylogic as fuzzy

from fuzzylogic.classes import Domain
import fuzzylogic.functions as ff

DEGTORAD = math.pi/180.0
RADTODEG = 180.0/math.pi

# Defining the main fuzzy domains
Ang = Domain("diff_theta", -180, 180, res=1)
Ang.zero = ff.triangular(-3,3)

Ang.p_small = ff.trapezoid(0, 3, 5, 10)
Ang.n_small = ff.trapezoid(-10,-5,-3,0)

Ang.small = Ang.p_small | Ang.n_small

Ang.p_med = ff.trapezoid(5, 10, 40, 50)
Ang.n_med = ff.trapezoid(-50,-40,-10,-5)

Ang.med = Ang.p_med | Ang.n_med

Ang.p_big = ff.trapezoid(40, 50, 80, 100)
Ang.n_big = ff.trapezoid(-100, -80, -50, -40)

Ang.big = Ang.p_big | Ang.n_big

Ang.p_large = ff.trapezoid(80, 100, 180, 190)
Ang.n_large = ff.trapezoid(-190, -180, -100, -80)

Ang.large = Ang.p_large | Ang.n_large

Dist = Domain("distance", 0, 8.0, res =0.05)
Dist.zero = ff.S(0, 0.05)
Dist.small = ff.trapezoid(0, 0.05, 0.2, 0.25)
Dist.med = ff.trapezoid(0.2, 0.25, 0.9, 1.1)
Dist.big = ff.trapezoid(0.9, 1.1, 3.5, 4.5)
Dist.large = ff.trapezoid(3.5, 4.5, 8.0, 9.0)


# reset_reason
NONE = 0
GAME_START = 1
SCORE_MYTEAM = 2
SCORE_OPPONENT = 3
GAME_END = 4
DEADLOCK = 5
GOALKICK = 6
CORNERKICK = 7
PENALTYKICK = 8
HALFTIME = 9
EPISODE_END = 10

# game_state
STATE_DEFAULT = 0
STATE_KICKOFF = 1
STATE_GOALKICK = 2
STATE_CORNERKICK = 3
STATE_PENALTYKICK = 4

# coordinates
MY_TEAM = 0
OP_TEAM = 1
BALL = 2
X = 0
Y = 1
TH = 2
ACTIVE = 3
TOUCH = 4


class Received_Image(object):
    def __init__(self, resolution, colorChannels):
        self.resolution = resolution
        self.colorChannels = colorChannels
        # need to initialize the matrix at timestep 0
        # rows, columns, colorchannels
        self.ImageBuffer = np.zeros(
            (resolution[1], resolution[0], colorChannels))

    def update_image(self, received_parts):
        self.received_parts = received_parts
        for i in range(0, len(received_parts)):
            # decode the base64 message
            dec_msg = base64.b64decode(self.received_parts[i].b64, '-_')
            # convert byte array to numpy array
            np_msg = np.fromstring(dec_msg, dtype=np.uint8)
            reshaped_msg = np_msg.reshape(
                (self.received_parts[i].height, self.received_parts[i].width, 3))
            for j in range(0, self.received_parts[i].height):  # y axis
                for k in range(0, self.received_parts[i].width):  # x axis
                    self.ImageBuffer[j + self.received_parts[i].y, k + self.received_parts[i].x, 0] = reshaped_msg[
                        j, k, 0]  # blue channel
                    self.ImageBuffer[j + self.received_parts[i].y, k + self.received_parts[i].x, 1] = reshaped_msg[
                        j, k, 1]  # green channel
                    self.ImageBuffer[j + self.received_parts[i].y, k + self.received_parts[i].x, 2] = reshaped_msg[
                        j, k, 2]  # red channel


class SubImage(object):
    def __init__(self, x, y, width, height, b64):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.b64 = b64


class Frame(object):
    def __init__(self):
        self.time = None
        self.score = None
        self.reset_reason = None
        self.game_state = None
        self.subimages = None
        self.coordinates = None
        self.half_passed = None


class Component(ApplicationSession):
    """
    AI Base + Rule Based Algorithm
    """

    def __init__(self, config):
        ApplicationSession.__init__(self, config)

    def printConsole(self, message):
        print(message)
        sys.__stdout__.flush()

    def onConnect(self):
        self.join(self.config.realm)

    @inlineCallbacks
    def onJoin(self, details):

        ##############################################################################
        def init_variables(self, info):
            # Here you have the information of the game (virtual init() in random_walk.cpp)
            # List: game_time, number_of_robots
            #       field, goal, penalty_area, goal_area, resolution Dimension: [x, y]
            #       ball_radius, ball_mass,
            #       robot_size, robot_height, axle_length, robot_body_mass, ID: [0, 1, 2, 3, 4]
            #       wheel_radius, wheel_mass, ID: [0, 1, 2, 3, 4]
            #       max_linear_velocity, max_torque, codewords, ID: [0, 1, 2, 3, 4]
            self.game_time = info['game_time']
            self.number_of_robots = info['number_of_robots']

            self.field = info['field']
            self.goal = info['goal']
            self.penalty_area = info['penalty_area']
            # self.goal_area = info['goal_area']
            self.resolution = info['resolution']

            self.ball_radius = info['ball_radius']
            # self.ball_mass = info['ball_mass']

            self.robot_size = info['robot_size']
            # self.robot_height = info['robot_height']
            self.axle_length = info['axle_length']
            # self.robot_body_mass = info['robot_body_mass']

            # self.wheel_radius = info['wheel_radius']
            # self.wheel_mass = info['wheel_mass']

            self.max_linear_velocity = info['max_linear_velocity']
            # self.max_torque = info['max_torque']
            # self.codewords = info['codewords']

            self.colorChannels = 3
            self.end_of_frame = False
            self.image = Received_Image(self.resolution, self.colorChannels)
            self.cur_posture = []
            self.cur_ball = []
            self.prev_posture = []
            self.prev_ball = []
            self.previous_frame = Frame()
            self.received_frame = Frame()

            self.cur_count = 0
            self.end_count = 0
            self.prev_sender = None
            self.sender = None
            self.touch = [False, False, False, False, False]
            self.prev_receiver = None
            self.receiver = None
            self.def_idx = 0
            self.atk_idx = 0
            self.closest_order = []
            self.player_state = [None, None, None, None, None]

            self.wheels = [0 for _ in range(10)]
            return

        ##############################################################################

        try:
            info = yield self.call(u'aiwc.get_info', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            try:
                self.sub = yield self.subscribe(self.on_event, args.key)
            except Exception as e2:
                self.printConsole("Error: {}".format(e2))

        init_variables(self, info)

        try:
            yield self.call(u'aiwc.ready', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            self.printConsole("I am ready for the game!")

    # set the left and right wheel velocities of robot with id 'id'
    # 'max_velocity' scales the velocities up to the point where at least one of wheel is operating at max velocity
    def set_wheel_velocity(self, id, left_wheel, right_wheel, max_velocity = False):
        multiplier = 1

        # wheel velocities need to be scaled so that none of wheels exceed the maximum velocity available
        # otherwise, the velocity above the limit will be set to the max velocity by the simulation program
        # if that happens, the velocity ratio between left and right wheels will be changed that the robot may not execute
        # turning actions correctly.
        if (abs(left_wheel) > self.max_linear_velocity[id] or abs(right_wheel) > self.max_linear_velocity[id] or max_velocity):
            if (abs(left_wheel) > abs(right_wheel)):
                multiplier = self.max_linear_velocity[id] / abs(left_wheel)
            else:
                multiplier = self.max_linear_velocity[id] / abs(right_wheel)

        self.wheels[2 * id] = left_wheel * multiplier
        self.wheels[2 * id + 1] = right_wheel * multiplier

    # let the robot with id 'id' move to a target position (x, y)
    # the trajectory to reach the target position is determined by several different parameters
    def set_target_position(self, id, x, y, scale, mult_lin, mult_ang, max_velocity):
        damping = 0.35
        ka = 0
        sign = 1

        # calculate how far the target position is from the robot
        dx = x - self.cur_posture[id][X]
        dy = y - self.cur_posture[id][Y]
        d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

        # calculate how much the direction is off
        desired_th = (math.pi / 2) if (dx == 0 and dy ==
                                       0) else math.atan2(dy, dx)
        d_th = desired_th - self.cur_posture[id][TH]
        while (d_th > math.pi):
            d_th -= 2 * math.pi
        while (d_th < -math.pi):
            d_th += 2 * math.pi

        # based on how far the target position is, set a parameter that
        # decides how much importance should be put into changing directions
        # farther the target is, less need to change directions fastly
        if (d_e > 1):
            ka = 17 / 90
        elif (d_e > 0.5):
            ka = 19 / 90
        elif (d_e > 0.3):
            ka = 21 / 90
        elif (d_e > 0.2):
            ka = 23 / 90
        else:
            ka = 25 / 90

        # if the target position is at rear of the robot, drive backward instead
        if (d_th > helper.d2r(95)):
            d_th -= math.pi
            sign = -1
        elif (d_th < helper.d2r(-95)):
            d_th += math.pi
            sign = -1

        # if the direction is off by more than 85 degrees,
        # make a turn first instead of start moving toward the target
        if (abs(d_th) > helper.d2r(85)):
            self.set_wheel_velocity(
                id, -mult_ang * d_th, mult_ang * d_th, False)
        # otherwise
        else:
            # scale the angular velocity further down if the direction is off by less than 40 degrees
            if (d_e < 5 and abs(d_th) < helper.d2r(40)):
                ka = 0.1
            ka *= 4

            # set the wheel velocity
            # 'sign' determines the direction [forward, backward]
            # 'scale' scales the overall velocity at which the robot is driving
            # 'mult_lin' scales the linear velocity at which the robot is driving
            # larger distance 'd_e' scales the base linear velocity higher
            # 'damping' slows the linear velocity down
            # 'mult_ang' and 'ka' scales the angular velocity at which the robot is driving
            # larger angular difference 'd_th' scales the base angular velocity higher
            # if 'max_velocity' is true, the overall velocity is scaled to the point
            # where at least one wheel is operating at maximum velocity
            self.set_wheel_velocity(id,
                                    sign * scale * (mult_lin * (
                                        1 / (1 + math.exp(-3 * d_e)) - damping) - mult_ang * ka * d_th),
                                    sign * scale * (mult_lin * (
                                        1 / (1 + math.exp(-3 * d_e)) - damping) + mult_ang * ka * d_th),
                                    max_velocity)

    # Turn right in a circle with radius at speed
    def speed_turn_circle(self, id, radius, speed=1.0):

        if speed == 0.0:
            speed = 0.001

        half_axle = self.axle_length[id]/2.0

        v_right = (1 - half_axle / radius) * speed
        v_left  = (1 + half_axle / radius) * speed

        if radius == 0:
            v_right = speed
            v_left  = speed

        return(v_left, v_right)
    

    def move_turn_circle(self, id, radius, speed=1.0, max_velocity=False):
        
        v_left, v_right = self.speed_turn_circle(id, radius, speed, max_velocity)

        self.set_wheel_velocity(id, v_left, v_right, max_velocity)

    def limit_speed_to_ang_vel(self, id, v_left, v_right, ang_vel):
        if (abs(v_left - v_right)/self.axle_length[id]) > ang_vel:
            div = abs(v_left - v_right) / (self.axle_length[id] * ang_vel)
            v_left /= div
            v_right /= div
        
        return(v_left, v_right)

    # Moves to given coordinates at a given top-speed. Uses circle-turns if possible and does not care about robot-orientation
    def move_to_circles(self, id, x, y, radius=0.5, speed=1.0, goal_speed=0.0, goal_angle=None, max_velocity = False, kick=True, debug = False):
        if max_velocity:
            speed = self.max_linear_velocity[id]
        
        max_turn_d_th = 480*DEGTORAD
        min_turn_d_th = 90*DEGTORAD

        d_t = self.received_frame.time - self.previous_frame.time
        if d_t == 0.0:
            d_t = 0.2
        
        my_x = self.cur_posture[id][X]
        my_y = self.cur_posture[id][Y]

        last_x = self.prev_posture[id][X]
        last_y = self.prev_posture[id][Y]

        dist = helper.dist(my_x, x, my_y, y)
        last_dist = helper.dist(last_x, x, last_y, y)
        d_dist = (dist - last_dist)/d_t

        angle_to_target = self.direction_angle(id, x, y)
        my_angle = self.cur_posture[id][TH]
        last_angle = self.prev_posture[id][TH]

        diff_theta = helper.clamp(my_angle - angle_to_target) # right-facing angle to turn to target
        d_theta = (helper.clamp(my_angle - last_angle)) / d_t

        direction = 1.0
        if abs(diff_theta) > math.pi/2:  # drive backwards if easier
            direction = -1.0
            diff_theta = helper.clamp(diff_theta-math.pi)

        if goal_angle != None:
            assert type(goal_angle) == type(1.0) or type(goal_angle) == type(1)
            # move to the target position at a given orientation (Two turn-circles)

            dir_x = math.cos(goal_angle)
            dir_y = math.sin(goal_angle)

            bot_dir_x = math.cos(my_angle)
            bot_dir_y = math.sin(my_angle)

            max_corridor_dist = self.axle_length[id]/2 + math.sin(3*DEGTORAD)*dist # Cone shaped approach corridor

            tar_to_bot_x = my_x - x
            tar_to_bot_y = my_y - y

            intersect_dist_tar = (bot_dir_x*tar_to_bot_x + bot_dir_y*tar_to_bot_y) / (bot_dir_x*dir_x + bot_dir_y*dir_y)
            intersect_dist_bot = -(dir_x*tar_to_bot_x + dir_y*tar_to_bot_y) / (bot_dir_x*dir_x + bot_dir_y*dir_y)

            rel_angle = goal_angle - my_angle
            turn_radius = intersect_dist_tar * math.tan(rel_angle/2.0)

            if debug:
                print(f"int_d_ball: {intersect_dist_tar}")
                print(f"int_d_bot : {intersect_dist_bot}")
                print(f"rel_angle : {rel_angle}")
                print(f"turn_rad  : {turn_radius}")

            corridor_dist = dir_x*tar_to_bot_y - dir_y*tar_to_bot_x
            correct_direction = (dir_x*tar_to_bot_x + dir_y*tar_to_bot_y) >= 0.0

            if abs(corridor_dist) > max_corridor_dist or not correct_direction:
                # If outside the corridor
                x_int_goal = x + dir_x*radius
                y_int_goal = y + dir_y*radius

                int_dist = helper.dist(my_x, x_int_goal, my_y, y_int_goal)

                if int_dist > dist:
                    x_temp_goal = tar_to_bot_y * corridor_dist
                    y_temp_goal = -tar_to_bot_x * corridor_dist

                    l_temp = math.sqrt(x_temp_goal ** 2 + y_temp_goal ** 2)
                    x_temp_goal /= l_temp
                    y_temp_goal /= l_temp

                    x_temp_goal = x + x_temp_goal
                    y_temp_goal = y + y_temp_goal

                    self.move_to_circles(id, x_temp_goal, y_temp_goal, radius, speed, goal_speed=speed*0.99, max_velocity=max_velocity, debug=debug)
                    return
                else:
                    self.move_to_circles(id, x_int_goal, y_int_goal, radius, speed, goal_speed=speed*0.99, max_velocity=max_velocity, debug=debug)
                    return

                
        if ((abs(diff_theta)*RADTODEG < 2 and dist < radius) or (self.player_state[id] == 'kick')) and d_dist < 0.0 and goal_speed > speed:
            # If inside the corridor and close to the target
            # Kick
            #print("kicking")

            self.player_state[id] = 'kick'
            targ_speed = dist/radius * speed + (1-dist/radius) * goal_speed

            self.set_wheel_velocity(id, targ_speed*direction, targ_speed*direction)
            return
        else:
            self.player_state[id] = None

        # just move to the target position. Orientation is not important (Only one turn-circle)

        f_dist = Dist(dist)

        # Speed
        speed_far =    speed
        speed_small =  (speed + goal_speed)/2
        speed_zero =   goal_speed

        # Defuzzy Speed
        speed = f_dist['zero'] * speed_zero + f_dist['small'] * speed_small + (f_dist['med'] + f_dist['big'] + f_dist['large']) * speed_far

        # Fuzzy diff_theta
        # The fuzzy domain Ang is in ° for convenience

        f_d_theta = Ang(diff_theta*RADTODEG)

        # Use p-d control for the angular velocity
        if id == 0:
            p = 1
            d = 0.12
        elif id == 1 or id == 2:
            p = 1
            d = 0.12
        else:
            p = 1
            d = 0.12

        p_d = p*diff_theta + d*d_theta

        # In the turn-circle (meaning while diff_theta is not small)
        max_circ_radius = dist/4.0*math.cos(diff_theta)
        circ_rad = min(max_circ_radius, radius)
        #circ_rad = max_circ_radius
        if p_d < 0.0:
            circ_rad *= -1.0

        circ_l, circ_r = self.speed_turn_circle(id, circ_rad*direction, speed*direction)

        # On the straight (when diff_theta is small)
        straight_rad = 1/(3*math.sin(p_d))
        if p_d < 0.0:
            straight_rad *= -1.0

        stra_l, stra_r = self.speed_turn_circle(id, straight_rad*direction, speed*direction)

        # When angle is zero
        zero_l, zero_r = speed * direction, speed * direction

        # Defuzzy straight and circle radius
        used_l  = f_d_theta['zero'] * zero_l + f_d_theta['small'] * stra_l + (f_d_theta['med'] + f_d_theta['big'] +f_d_theta['large']) * circ_l
        used_r  = f_d_theta['zero'] * zero_r + f_d_theta['small'] * stra_r + (f_d_theta['med'] + f_d_theta['big'] +f_d_theta['large']) * circ_r

        used_l, used_r = self.limit_speed_to_ang_vel(id, used_l, used_r, min(max_turn_d_th, max(abs(8*diff_theta + 0.05*d_theta), min_turn_d_th)))

        used_d_th = (used_l - used_r) / self.axle_length[id]

        if debug:
            print("__________________________________________")
            print(f"id:{id}")
            print(f"diff_theta:{diff_theta*RADTODEG}")
            print(f"d_theta:{d_theta*RADTODEG}")
            print(f"th_zero:{f_d_theta['zero']}, th_small:{f_d_theta['small']}, th_big:{(f_d_theta['med'] + f_d_theta['big'] +f_d_theta['large'])}")
            print(f"dist:{dist}, d_dist:{d_dist}")
            print(f"d_zero:{f_dist['zero']}, d_small:{f_dist['small']}, d_big:{(f_dist['med'] + f_dist['big'] + f_dist['large'])}")
            print(f"speed:{speed}")
            print(f"used_l:{used_l}, used_r:{used_r}")
            print(f"used_speed:{(used_l + used_r)/2.0}")
            print(f"used_d_th:{used_d_th*RADTODEG}")
            print(f"direction:{direction}")

        self.set_wheel_velocity(id, used_l, used_r)


    # copy coordinates from frames to different variables just for convenience
    def get_coord(self):
        self.cur_ball = self.received_frame.coordinates[BALL]
        self.cur_posture = self.received_frame.coordinates[MY_TEAM]
        self.cur_posture_op = self.received_frame.coordinates[OP_TEAM]
        self.prev_ball = self.previous_frame.coordinates[BALL]
        self.prev_posture = self.previous_frame.coordinates[MY_TEAM]
        self.prev_posture_op = self.previous_frame.coordinates[OP_TEAM]

    # find a defender and a forward closest to the ball
    def find_closest_robot(self):
        # find the closest defender
        min_idx = 0
        min_dist = 9999.99
        def_dist = 9999.99

        all_dist = []

        for i in [1, 2]:
            measured_dist = helper.dist(self.cur_ball[X], self.cur_posture[i][X], self.cur_ball[Y],
                                        self.cur_posture[i][Y])
            all_dist.append(measured_dist)
            if (measured_dist < min_dist):
                min_dist = measured_dist
                def_dist = min_dist
                min_idx = i

        self.def_idx = min_idx

        # find the closest forward
        min_idx = 0
        min_dist = 9999.99
        atk_dist = 9999.99

        for i in [3, 4]:
            measured_dist = helper.dist(self.cur_ball[X], self.cur_posture[i][X], self.cur_ball[Y],
                                        self.cur_posture[i][Y])
            all_dist.append(measured_dist)
            if (measured_dist < min_dist):
                min_dist = measured_dist
                atk_dist = min_dist
                min_idx = i

        self.atk_idx = min_idx

        # record the robot closer to the ball between the two too
        self.closest_order = np.argsort(all_dist) + 1

    # predict where the ball will be located after 'steps' steps
    def predict_ball_location(self, steps):
        dx = self.cur_ball[X] - self.prev_ball[X]
        dy = self.cur_ball[Y] - self.prev_ball[Y]
        return [self.cur_ball[X] + steps * dx, self.cur_ball[Y] + steps * dy]

    # predict, where the ball will cross the given line (in y direction)
    def predict_goal_cross(self, x_keeper):
        dx = self.cur_ball[X] - self.prev_ball[X]
        dy = self.cur_ball[Y] - self.prev_ball[Y]

        if dx >= -0.005:
            y_res = 0
        else:
            mult = (self.cur_ball[X] - x_keeper) / dx
            y_res = self.cur_ball[Y] - dy * mult

        print(f"y_res:{y_res}")
        return(y_res)

    # let the robot face toward specific direction
    def face_specific_position(self, id, x, y):
        dx = x - self.cur_posture[id][X]
        dy = y - self.cur_posture[id][Y]

        desired_th = (math.pi / 2) if (dx == 0 and dy ==
                                       0) else math.atan2(dy, dx)

        self.angle(id, desired_th)

    # returns the angle toward a specific position from current robot posture
    def direction_angle(self, id, x, y):
        dx = x - self.cur_posture[id][X]
        dy = y - self.cur_posture[id][Y]

        return ((math.pi / 2) if (dx == 0 and dy == 0) else math.atan2(dy, dx))

    # turn to face 'desired_th' direction
    def angle(self, id, desired_th):
        mult_ang = 0.4

        d_th = desired_th - self.cur_posture[id][TH]
        d_th = helper.trim_radian(d_th)

        # the robot instead puts the direction rear if the angle difference is large
        if (d_th > helper.d2r(95)):
            d_th -= math.pi
            sign = -1
        elif (d_th < helper.d2r(-95)):
            d_th += math.pi
            sign = -1

        self.set_wheel_velocity(id, -mult_ang * d_th, mult_ang * d_th, False)

    # checks if a certain position is inside the penalty area of 'team'
    def in_penalty_area(self, obj, team):
        if (abs(obj[Y]) > self.penalty_area[Y] / 2):
            return False

        if (team == MY_TEAM):
            return (obj[X] < -self.field[X] / 2 + self.penalty_area[X])
        else:
            return (obj[X] > self.field[X] / 2 - self.penalty_area[X])

    # check if the ball is coming toward the robot
    def ball_coming_toward_robot(self, id):
        x_dir = abs(self.cur_posture[id][X] - self.prev_ball[X]
                    ) > abs(self.cur_posture[id][X] - self.cur_ball[X])
        y_dir = abs(self.cur_posture[id][Y] - self.prev_ball[Y]
                    ) > abs(self.cur_posture[id][Y] - self.cur_ball[Y])

        # ball is coming closer
        if (x_dir and y_dir):
            return True
        else:
            return False

    # check if the robot with id 'id' has a chance to shoot
    def shoot_chance(self, id):
        dx = self.cur_ball[X] - self.cur_posture[id][X]
        dy = self.cur_ball[Y] - self.cur_posture[id][Y]

        # if the ball is located further on left than the robot, it will be hard to shoot
        if (dx < 0):
            return False

        # if the robot->ball direction aligns with opponent's goal, the robot can shoot
        y = (self.field[X] / 2 - self.cur_ball[X]) * \
            dy / dx + self.cur_posture[id][Y]
        if (abs(y) < self.goal[Y] / 2):
            return True
        else:
            return False

    # check if sender/receiver pair should be reset
    def reset_condition(self):
        # if the time is over, setting is reset
        if (self.end_count > 0 and self.end_count - self.cur_count < 0):
            return True

        # if there is no sender and receiver is not in shoot chance, setting is cleared
        if not self.sender is None:
            if not self.shoot_chance(self.sender):
                return True
        return False

    # check if a sender can be selected
    def set_sender_condition(self):
        for i in range(1, 5):
            # if this robot is near the ball, it will be a sender candidate
            dist = helper.dist(
                self.cur_posture[i][X], self.cur_ball[X], self.cur_posture[i][Y], self.cur_ball[Y])
            if dist < 0.5 and self.cur_posture[i][ACTIVE]:
                return True
        return False

    # check if a receiver should be selected
    def set_receiver_condition(self):
        # if a sender exists, any other robots can be receiver candidates
        if self.sender != None and self.receiver == None:
            return True
        return False

    # select a sender
    def set_sender(self, _player_list):
        distance_list = []
        for sender in _player_list:
            predict_ball = self.predict_ball_location(3)
            ball_distance = helper.dist(
                predict_ball[X], self.cur_posture[sender][X], predict_ball[Y], self.cur_posture[sender][Y])
            distance_list.append(ball_distance)

        # if the distance between ball and sender is less than 1, choose the closest robot as the sender
        if min(distance_list) < 1.0:
            return distance_list.index(min(distance_list)) + 1

        # otherwise, there is no sender
        return None

    # select a receiver
    def set_receiver(self, _player_list):
        receiver_op_dist_list = []
        for receiver in _player_list:
            temp_receiver_op_dist_list = []
            # the sender is not a receiver candidate
            if receiver == self.sender:
                receiver_op_dist_list.append(999)
                continue

            # the distance between the robot and opponents
            for op in range(1, 5):  # [1,2,3,4]
                op_distance = helper.dist(
                    self.cur_posture[receiver][X], self.cur_posture_op[op][X], self.cur_posture[receiver][Y], self.cur_posture_op[op][Y])
                temp_receiver_op_dist_list.append(op_distance)

            # save the shortest distance between this robot and one of opponents
            receiver_op_dist_list.append(min(temp_receiver_op_dist_list))

        receiver_ball_list = []
        for r in receiver_op_dist_list:
            # if the minimum distance between player and opponent's player is less than 0.5, this robot cannot be receiver
            if r < 0.5 or r == 999:
                receiver_ball_list.append(999)
                continue
            id = receiver_op_dist_list.index(r) + 1
            receiver_ball_distance = helper.dist(
                self.cur_ball[X], self.cur_posture[id][X], self.cur_ball[Y], self.cur_posture[id][Y])
            receiver_ball_list.append(receiver_ball_distance)

        if min(receiver_ball_list) < 999:
            min_id = receiver_ball_list.index(min(receiver_ball_list)) + 1
            return min_id
        return None

    def pass_ball(self):
        # and not None in [self.prev_sender, self.prev_receiver, self.sender, self.receiver] :
        if self.prev_sender == self.receiver or self.prev_receiver == self.sender:
            self.sender = self.prev_sender
            self.receiver = self.prev_receiver

        self.receive_ball()
        self.send_ball()

        self.prev_sender = self.sender
        self.prev_receiver = self.receiver

    def send_ball(self):
        if self.sender == None:
            return

        goal_dist = helper.dist(
            4.0, self.cur_posture[self.sender][X], 0, self.cur_posture[self.sender][Y])
        # if the sender has a shoot chance, it tries to shoot
        if self.shoot_chance(self.sender):
            if goal_dist > 0.3 * self.field[X] / 2:
                self.actions(self.sender, 'dribble', refine=True)
                return
            else:
                self.actions(self.sender, 'kick')
                return

        # if the receiver exists, get the distance between the sender and the receiver
        sender_receiver_dist = None
        if not self.receiver == None:
            sender_receiver_dist = helper.dist(
                self.cur_posture[self.sender][X], self.cur_posture[self.receiver][X], self.cur_posture[self.sender][Y], self.cur_posture[self.receiver][Y])

        # if the sender is close to the receiver, the sender kicks the ball
        if not sender_receiver_dist == None:
            if sender_receiver_dist < 0.3 and not self.cur_posture[self.receiver][TOUCH]:
                self.actions(self.sender, 'kick')
                return

        ift, theta_diff = self.is_facing_target(
            self.sender, self.cur_ball[X], self.cur_ball[Y])
        if not ift:
            # after the sender kicks, it stops
            if theta_diff > math.pi * 3/4:
                self.actions(self.sender, None)
                return
            else:
                self.actions(self.sender, 'follow', refine=True)
                return

        # if the ball is in front of the sender and sender is moving backward
        if self.cur_posture[self.sender][X] < - 0.8 * self.field[X] / 2:
            if self.cur_posture[self.sender][X] - self.prev_posture[self.sender][X] < 0:
                self.actions(self.sender, 'backward')

        self.actions(self.sender, 'dribble', refine=True)
        return

    def receive_ball(self):
        # if receiver does not exist, do nothing
        if self.receiver == None:
            return

        goal_dist = helper.dist(
            4.0, self.cur_posture[self.receiver][X], 0, self.cur_posture[self.receiver][Y])
        # if sender is in shoot chance, receiver does nothing(reset)
        if self.shoot_chance(self.sender):
            self.actions(self.receiver, None)
            return
        # if receiver is in shoot chance, receiver try to shoot
        if self.shoot_chance(self.receiver):
            if goal_dist > 0.3 * self.field[X] / 2:
                self.actions(self.receiver, 'dribble', refine=True)
                return
            else:
                self.actions(self.receiver, 'kick')
                return

        # if sender exists
        if not self.sender == None:
            s2risFace, _ = self.is_facing_target(
                self.sender, self.cur_posture[self.receiver][X], self.cur_posture[self.receiver][Y], 4)
            r2sisFace, _ = self.is_facing_target(
                self.receiver, self.cur_posture[self.sender][X], self.cur_posture[self.sender][Y], 4)
            # if sender and receiver directs each other
            if s2risFace and r2sisFace:
                if self.cur_posture[self.receiver][TH] > 0 or self.cur_posture[self.receiver][TH] < -3:
                    self.actions(self.receiver, 'follow', [
                                 self.prev_posture[self.receiver][X], self.prev_posture[self.receiver][Y] - 0.5 * self.field[Y]])
                    return
                self.actions(self.receiver, 'follow', [
                             self.prev_posture[self.receiver][X], self.prev_posture[self.receiver][Y] + 0.5 * self.field[Y]])
                return

        r_point = self.cur_ball
        # if sender exists
        if not self.sender == None:
            r_point = self.receive_position()
        receiver_ball_dist = helper.dist(
            self.cur_ball[X], self.cur_posture[self.receiver][X], self.cur_ball[Y], self.cur_posture[self.receiver][Y])
        # if ball is close to receiver
        if receiver_ball_dist > 0.3 * self.field[X] / 2:
            self.actions(self.receiver, 'follow', [
                         r_point[X], r_point[Y]], refine=True)
            return

        r2bisFace, _ = self.is_facing_target(
            self.receiver, self.cur_ball[X], self.cur_ball[Y], 4)
        if not r2bisFace:
            self.actions(self.receiver, 'follow', refine=True)
            return
        # if receiver is moving to our goal area
        if self.cur_posture[self.receiver][X] < - 0.8 * self.field[X] / 2:
            if self.cur_posture[self.receiver][X] - self.prev_posture[self.receiver][X] < 0:
                self.actions(self.receiver, 'backward')

        self.actions(self.receiver, 'dribble')
        return

    # let robot with id 'id' execute an action directed by 'mode'
    def actions(self, id, mode=None, target_pts=None, params=None, refine=False):
        if id == None:
            return

        # if the player state is set to 'stop', force the mode to be 'stop'
        if self.player_state[id] == 'stop':
            mode = 'stop'

        if mode == None:
            # reset all robot status
            if self.sender == id:
                self.sender = None
                self.touch = [False, False, False, False, False]
            if self.receiver == id:
                self.receiver = None
            self.player_state[id] = None
            return
        if mode == 'follow':
            # let the robot follow the ball
            if target_pts == None:
                target_pts = self.predict_ball_location(3)
            if params == None:
                params = [1.0, 3.0, 0.6, False]
            if refine:
                self.set_pos_parameters(id, target_pts, params)
            self.set_target_position(
                id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            self.player_state[id] = 'follow'
            return
        if mode == 'dribble':
            # let the robot follow the ball but at a faster speed
            if target_pts == None:
                target_pts = self.cur_ball
            if params == None:
                params = [1.4, 5.0, 0.8, False]
            if refine:
                self.set_pos_parameters(id, target_pts, params)
            self.set_target_position(
                id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            self.player_state[id] = 'dribble'
            return
        if mode == 'kick':
            # kick the ball
            if target_pts == None:
                target_pts = self.cur_ball
            if params == None:
                params = [1.4, 5.0, 0.8, True]
            if self.end_count == 0 and not self.touch[id]:
                self.end_count = self.cur_count + 10  # 0.05 * cnt seconds
            self.player_state[id] = 'kick'
            if self.touch[id]:
                self.player_state[id] = 'stop'
            if not self.touch[id]:
                self.touch[id] = self.cur_posture[id][TOUCH]
            if self.player_state[id] == 'stop':
                params = [0.0, 0.0, 0.0, False]
            self.set_target_position(
                id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            return
        if mode == 'stop':
            # stop while counter is on
            if params == None:
                params = [0.0, 0.0, False]
            self.set_wheel_velocity(id, params[0], params[1], params[2])
            if self.end_count == 0:
                self.end_count = self.cur_count + 5  # 0.05 * cnt seconds
            self.player_state[id] = 'stop'
            if self.end_count - 1 == self.cur_count:
                self.player_state[id] = None
            return
        if mode == 'backward':
            # retreat from the current position
            if target_pts == None:
                target_pts = [self.cur_posture[id]
                              [X] + 0.2, self.cur_posture[id][Y]]
            if params == None:
                params = [1.4, 5.0, 0.8, False]
            if refine:
                self.set_pos_parameters(id, target_pts, params)
            self.set_target_position(
                id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            self.player_state[id] = 'backward'
            return
        if mode == 'position':
            # go toward target position
            self.set_target_position(
                id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            return

    def set_pos_parameters(self, id, target_pts, params, mult=1.2):
        prev_dist = helper.dist(
            self.prev_posture[id][X], target_pts[X], self.prev_posture[id][Y], target_pts[Y])
        cur_dist = helper.dist(
            self.cur_posture[id][X], target_pts[X], self.cur_posture[id][Y], target_pts[Y])
        if cur_dist > prev_dist - 0.02:
            params = [params[0] * mult, params[1]
                      * mult, params[2] * mult, params[3]]
        return params

    def is_facing_target(self, id, x, y, div=4):
        dx = x - self.cur_posture[id][X]
        dy = y - self.cur_posture[id][Y]
        ds = math.sqrt(dx * dx + dy * dy)
        desired_th = (self.cur_posture[id][TH] if (
            ds == 0) else math.acos(dx / ds))

        theta = self.cur_posture[id][TH]
        if desired_th < 0:
            desired_th += math.pi * 2
        if theta < 0:
            theta += math.pi * 2
        diff_theta = abs(desired_th - theta)
        if diff_theta > math.pi:
            diff_theta = min(diff_theta, math.pi * 2 - diff_theta)
        if diff_theta < math.pi / div or diff_theta > math.pi * (1 - 1 / div):
            return [True, diff_theta]
        return [False, diff_theta]

    def receive_position(self):
        step = 5
        ball_receiver_dist = helper.dist(self.cur_ball[X], self.cur_posture[self.receiver][X], self.cur_ball[Y],
                                         self.cur_posture[self.receiver][Y])
        prev_ball_receiver_dist = helper.dist(self.prev_ball[X], self.prev_posture[self.receiver][X],
                                              self.prev_ball[Y], self.prev_posture[self.receiver][Y])

        diff_dist = prev_ball_receiver_dist - ball_receiver_dist
        if diff_dist > 0:
            step = ball_receiver_dist  # diff_dist

        step = min(step, 15)

        predict_pass_point = self.predict_ball_location(step)

        ball_goal_dist = helper.dist(
            self.cur_ball[X], self.field[X] / 2, self.cur_ball[Y], 0)
        prev_ball_goal_dist = helper.dist(
            self.prev_ball[X], self.field[X] / 2, self.prev_ball[Y], 0)
        if ball_goal_dist > prev_ball_goal_dist:
            predict_pass_point[X] = predict_pass_point[X] - 0.15

        return predict_pass_point

    @inlineCallbacks
    def on_event(self, f):

        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return

        # Move the fuck away
        def gtfo(self, id):
            if id == 0:
                self.move_to_circles(
                    id, -self.field[X]/2. * 0.9, self.field[Y]/2. * 0.9, goal_speed=3.0, max_velocity=False)
            elif id == 1:
                self.move_to_circles(
                    id, -self.field[X]/2. * 0.7, self.field[Y]/2. * 0.9, goal_speed=3.0, max_velocity=False)
            elif id == 2:
                self.move_to_circles(
                    id, -self.field[X]/2. * 0.9, -self.field[Y]/2. * 0.9, goal_speed=3.0, max_velocity=False)
            elif id == 3:
                self.move_to_circles(
                    id, -self.field[X]/2. * 0.5, self.field[Y]/2. * 0.9, goal_speed=3.0, max_velocity=False)
            else:
                self.move_to_circles(
                    id, -self.field[X]/2. * 0.7, -self.field[Y]/2. * 0.9, goal_speed=3.0, max_velocity=False)

        # initiate empty frame
        if (self.end_of_frame):
            self.received_frame = Frame()
            self.end_of_frame = False
        received_subimages = []

        if 'time' in f:
            self.received_frame.time = f['time']
        if 'score' in f:
            self.received_frame.score = f['score']
        if 'reset_reason' in f:
            self.received_frame.reset_reason = f['reset_reason']
        if 'game_state' in f:
            self.received_frame.game_state = f['game_state']
        if 'ball_ownership' in f:
            self.received_frame.ball_ownership = f['ball_ownership']
        if 'half_passed' in f:
            self.received_frame.half_passed = f['half_passed']
        if 'subimages' in f:
            self.received_frame.subimages = f['subimages']
            for s in self.received_frame.subimages:
                received_subimages.append(SubImage(s['x'],
                                                   s['y'],
                                                   s['w'],
                                                   s['h'],
                                                   s['base64'].encode('utf8')))
            self.image.update_image(received_subimages)
        if 'coordinates' in f:
            self.received_frame.coordinates = f['coordinates']
        if 'EOF' in f:
            self.end_of_frame = f['EOF']

        if (self.end_of_frame):
            # to get the image at the end of each frame use the variable:
            # self.image.ImageBuffer

            if (self.received_frame.reset_reason != NONE):
                self.previous_frame = copy.deepcopy(self.received_frame)

            self.get_coord()
            self.find_closest_robot()

            goal_x = -self.field[X]/2.0 + 0.25

            pred_y = self.predict_goal_cross(goal_x)

            goal_y = max(-0.6, min(0.6, pred_y))

            ##############################################################################
            if (self.received_frame.game_state == STATE_DEFAULT):
                # robot functions in STATE_DEFAULT
                for i in range(1,5):
                    gtfo(self, i)

                id = 0id
                self.move_to_circles(id, goal_x, goal_y, speed=1.5, goal_speed=0, max_velocity=True, debug=False)
                
                ang_goal_to_ball = math.atan(self.cur_ball[Y]/ (self.cur_ball[X]-self.field[X]/2.0))
                print(ang_goal_to_ball)

                for id in [1]:
                    self.move_to_circles(id, self.cur_ball[X], self.cur_ball[Y], speed=2, goal_speed=3, goal_angle=math.pi+ang_goal_to_ball)

                set_wheel(self, self.wheels)

            ##############################################################################
            if (self.received_frame.reset_reason == GAME_END):
                # (virtual finish() in random_walk.cpp)
                # save your data
                with open(args.datapath + '/result.txt', 'w') as output:
                    # output.write('yourvariables')
                    output.close()
                # unsubscribe; reset or leave
                yield self.sub.unsubscribe()
                try:
                    yield self.leave()
                except Exception as e:
                    self.printConsole("Error: {}".format(e))
            ##############################################################################

            self.end_of_frame = False
            self.previous_frame = copy.deepcopy(self.received_frame)

    def onDisconnect(self):
        if reactor.running:
            reactor.stop()


if __name__ == '__main__':

    try:
        unicode
    except NameError:
        # Define 'unicode' for Python 3
        def unicode(s, *_):
            return s

    def to_unicode(s):
        return unicode(s, "utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("server_ip", type=to_unicode)
    parser.add_argument("port", type=to_unicode)
    parser.add_argument("realm", type=to_unicode)
    parser.add_argument("key", type=to_unicode)
    parser.add_argument("datapath", type=to_unicode)

    args = parser.parse_args()

    ai_sv = "rs://" + args.server_ip + ":" + args.port
    ai_realm = args.realm

    # create a Wamp session object
    session = Component(ComponentConfig(ai_realm, {}))

    # initialize the msgpack serializer
    serializer = MsgPackSerializer()

    # use Wamp-over-rawsocket
    runner = ApplicationRunner(ai_sv, ai_realm, serializers=[serializer])

    runner.run(session, auto_reconnect=False)
