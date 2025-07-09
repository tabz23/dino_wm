import math
import random

import carla
import numpy as np
from .toolkit.carla_manager.utils import FollowDirections

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_location_distance, get_vehicle_orientation, get_vehicle_pos, get_vehicle_velocity


class CarlaFollowEnv(CarlaWptEnv):
    """
    Vehicle follows the car in front of it at a reasonable distance for going straight, turning left, and turning right.
    **Provided Tasks**: ``carla_follow``
    """
    def __init__(self, config):
        super().__init__(config)
        self.reset_to_state = None

    def state(self):
        """
        The state of the environment is the current position and orientation of the ego vehicle.
        """
        ego_tf = self.ego.get_transform()
        nonego_tf = self.nonego.get_transform()
        return [
            ego_tf.location.x, ego_tf.location.y, ego_tf.location.z, ego_tf.rotation.pitch, ego_tf.rotation.yaw, ego_tf.rotation.roll,
            nonego_tf.location.x, nonego_tf.location.y, nonego_tf.location.z, ego_tf.rotation.pitch, nonego_tf.rotation.yaw, nonego_tf.rotation.roll
        ]

    # def set_reset_to_state(self, reset_to_state):
    #     """
    #     Set the state to reset the environment to.
    #     The state should be a list of two lists, each containing the position and orientation of the ego and non-ego vehicles.
    #     """
    #     self.reset_to_state = reset_to_state
    # def on_reset(self) -> None:
    #     # checks which path the vehicle will take this time
    #     max_num_of_directions = len(self._config.lane_start_points)
    #     self.random_num = 0

    #     if self._config.direction == FollowDirections.RANDOM.value:
    #         self.random_num = random.randint(0, max_num_of_directions - 1)  # random path
    #     elif 0 <= self._config.direction < max_num_of_directions:
    #         self.random_num = self._config.direction  # set path

    #     # spawn the nonego vehicle
    #     self.nonego_spawn_point = self._config.nonego_spawn_points[self.random_num]
    #     nonego_transform = carla.Transform(carla.Location(*self.nonego_spawn_point[:3]), carla.Rotation(yaw=self.nonego_spawn_point[4]))
    #     # self.nonego = self._world.spawn_actor(transform=nonego_transform)
    #     self.nonego = self._world.spawn_actor()

    #     # spawn the ego vehicle
    #     self.ego_src = self._config.lane_start_points[self.random_num]
    #     ego_transform = carla.Transform(carla.Location(*self.ego_src[:3]), carla.Rotation(yaw=self.nonego_spawn_point[4]))
    #     # self.ego = self._world.spawn_actor(transform=ego_transform)
    #     self.ego = self._world.spawn_actor()

    #     # initializes all the values needed for the waypoint and velocity tracker and pid_controller
    #     self.prev_errors = {"last_error": 0.0, "integral": 0.0}
    #     self.nonego_direction = self.nonego_spawn_point[4]
    #     self.list_waypoints = []
    #     self.list_velocity = []

    #     # plans the path of the nonego vehicle
    #     nonego_dest = self._config.lane_end_points[self.random_num]
    #     dest_location = carla.Location(x=nonego_dest[0], y=nonego_dest[1], z=nonego_dest[2])
    #     self.nonego_planner = FixedEndingPlanner(self.nonego, dest_location)
    #     self.on_step()

    def on_reset(self) -> None:
        self.random_num = 0
        if self.reset_to_state is None:
            # Get a random valid waypoint on the driving lane for the ego vehicle
            all_waypoints = self._world._get_map().generate_waypoints(distance=2.0)
            driving_waypoints = [wp for wp in all_waypoints if wp.lane_type == carla.LaneType.Driving]
            ego_wp = random.choice(driving_waypoints)

            # Spawn the ego vehicle at this waypoint
            ego_transform = ego_wp.transform

            # Find a waypoint ahead in the same lane for the non-ego vehicle
            distance_ahead = 10.0  # meters ahead of ego
            forward_wps = ego_wp.next(distance_ahead)
            if not forward_wps:
                # fallback: just reset again to avoid failure
                return self.on_reset()

            nonego_wp = forward_wps[0]
            nonego_transform = nonego_wp.transform
        else:
            # Reset to a specific state
            ego_transform = carla.Transform(carla.Location(*self.reset_to_state[0][:3]), carla.Rotation(*self.reset_to_state[0][3:6]))
            nonego_transform = carla.Transform(carla.Location(*self.reset_to_state[1][:3]), carla.Rotation(*self.reset_to_state[1][3:6]))

        ego_transform.location.z += 0.5
        nonego_transform.location.z += 0.5

        self.ego = self._world.spawn_actor(transform=ego_transform)
        self.nonego = self._world.spawn_actor(transform=nonego_transform)

        if self.nonego is None or self.ego is None:
            return self.on_reset()  # Reset if spawning failed
        

        # Initialize error tracking for the PID controller
        self.prev_errors = {"last_error": 0.0, "integral": 0.0}
        self.nonego_direction = nonego_transform.rotation.yaw
        self.list_waypoints = []
        self.list_velocity = []

        # Plan a fixed destination far ahead (optional – here: 30m further)
        nonego_wp = self._world._get_map().get_waypoint(nonego_transform.location, project_to_road=True)
        if nonego_wp:
            planner_wp = nonego_wp.next(30.0)
            if planner_wp:
                dest_location = planner_wp[0].transform.location
                self.nonego_planner = FixedEndingPlanner(self.nonego, dest_location)

        # Take one step to initialize
        self.on_step()


    def on_step(self) -> None:
        # makes the nonego vehicle the destination of the ego vehicle
        nonego_x, nonego_y = get_vehicle_pos(self.nonego)
        dest_location = carla.Location(x=nonego_x, y=nonego_y, z=self._config.lane_end_points[self.random_num][2])
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()

        self.nonego_waypoints, self.nonego_planner_stats = self.nonego_planner.run_step()

        super().on_step()

    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        nonego_control = self.get_nonego_vehicle_control()
        self.ego.apply_control(control)
        self.nonego.apply_control(nonego_control)

    def get_nonego_vehicle_control(self):
        """
        Non-ego vehicle control is designed for the scenario.
        """
        nonego_loc = get_vehicle_pos(self.nonego)

        # Keep constant speed
        nonego_velocity = self.nonego.get_velocity()
        nonego_speed = math.sqrt((nonego_velocity.x) ** 2 + (nonego_velocity.y) ** 2)
        if abs(nonego_speed) < 2:
            acc = 2
        else:
            acc = 0

        # adds any new waypoints/next step to destination
        if len(self.nonego_waypoints) > 0:
            closest_waypoint = self.nonego_waypoints[0]

            # appends that new waypoint location into the waypoint tracker
            if closest_waypoint not in self.list_waypoints:
                self.list_waypoints.append(closest_waypoint)
                self.list_velocity.append(np.array([*get_vehicle_velocity(self.nonego)]))

            # changes the direction of the car based on the angle it makes with its x and y components with its current
            # location and its next location
            heading_angle = math.degrees(math.atan2(closest_waypoint[0] - nonego_loc[0], closest_waypoint[1] - nonego_loc[1]))
            if heading_angle < 90:
                heading_angle = 90 - heading_angle
            else:
                heading_angle = 450 - heading_angle

            heading_error = heading_angle - get_vehicle_orientation(self.nonego)
            if heading_error > 180:
                heading_error -= 360
            elif heading_error < -180:
                heading_error += 360

            coeffs = self._config.pid_coeffs
            self.control, self.prev_errors = self.pid_controller(heading_error, self.prev_errors, coeffs)

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), steer=np.clip(self.control, -1, 1), brake=float(brake))

    def pid_controller(self, error, prev_errors, coeffs):
        """
        Calculate the PID control output to minimize the deviation.
        Args:
        target (float): The target for the PID controller (central line x-coordinate).
        current (float): The current measurement of the process variable (vehicle x-coordinate).
        prev_errors (dict): A dictionary holding the last error and the integral of errors.
        coeffs (tuple): A tuple of PID coefficients (Kp, Ki, Kd).
        Returns:
        float: The control output (steering angle adjustment).
        dict: Updated dictionary with the last error and integral.
        """
        Kp, Ki, Kd = coeffs
        integral = prev_errors["integral"] + error
        derivative = error - prev_errors["last_error"]

        output = (Kp * error) + (Ki * integral) + (Kd * derivative)

        # Update the errors for the next call
        updated_errors = {"last_error": error, "integral": integral}

        return output, updated_errors

    def is_destination_reached(self):
        return len(self.nonego_waypoints) <= 1

    def reward(self):
        total_reward, info = super().reward()
        total_reward -= info["r_speed"] + info["r_waypoints"]
        del info["r_speed"]
        del info["r_waypoints"]

        reward_scales = self._config.reward.scales

        # gets the starting distance between the two vehicles
        original_dist = get_location_distance(
            (self._config.lane_start_points[self.random_num][0], self._config.lane_start_points[self.random_num][1]),
            (self._config.nonego_spawn_points[self.random_num][0], self._config.nonego_spawn_points[self.random_num][1]),
        )
        # gets the ending distance between the two vehicles
        current_dist = get_location_distance(get_vehicle_pos(self.ego), get_vehicle_pos(self.nonego))

        # if the current distance between the two vehicles is in an acceptable range, give reward
        if current_dist < original_dist + 2 and current_dist >= original_dist:
            p_dist = 0.2 * reward_scales["distance"]
        else:
            p_dist = -abs(current_dist - original_dist) * reward_scales["distance"]

        # check to make sure the ego vehicle is following the correct waypoints, speed, and direction of where the
        # nonego vehicle was at that time
        ego_velocity = np.array([*get_vehicle_velocity(self.ego)])
        if np.array_equal(ego_velocity, self.list_velocity[0]):
            p_velocity = 0.2 * reward_scales["velocity"]
        else:
            p_velocity = 0

        if get_vehicle_pos(self.ego) == self.list_waypoints[0]:
            p_waypoints = 0.2 * reward_scales["waypoints"]
            self.list_velocity.pop(0)
            self.list_waypoints.pop(0)
        else:
            p_waypoints = 0

        total_reward += p_dist + p_waypoints + p_velocity
        return total_reward, info

    def get_terminal_conditions(self):
        terminal_config = self._config.terminal
        info = super().get_terminal_conditions()

        # if the distance between the two vehicles is too long, reset the scenario
        ego_x, ego_y = get_vehicle_pos(self.ego)
        nonego_loc = self.nonego.get_transform().location
        dist = math.sqrt((ego_x - nonego_loc.x) ** 2 + (ego_y - nonego_loc.y) ** 2)
        if dist > terminal_config.terminal_dist:
            info["terminal_dist"] = True

        return info
    
    def compute_continuous_action(self):
        # === Ego vehicle state ===
        ego_pos = get_vehicle_pos(self.ego)
        ego_yaw_rad = math.radians(self.ego.get_transform().rotation.yaw)
        ego_speed = np.linalg.norm(get_vehicle_velocity(self.ego))

        # === Leader (nonego) vehicle state ===
        leader_location = self.nonego.get_location()
        leader_pos = (leader_location.x, leader_location.y)

        dx = leader_pos[0] - ego_pos[0]
        dy = leader_pos[1] - ego_pos[1]

        # === Desired yaw (toward leader) ===
        desired_yaw_rad = math.atan2(dy, dx)

        # === Heading error ===
        heading_error_rad = (desired_yaw_rad - ego_yaw_rad + np.pi) % (2 * np.pi) - np.pi

        # === Steering command ===
        Kp_steer = 0.75
        steer_cmd = -Kp_steer * heading_error_rad
        steer_cmd = np.clip(steer_cmd, -1.0, 1.0)

        # === Distance control ===
        distance_to_leader = np.linalg.norm([dx, dy])
        desired_distance = 8.0

        # **Inject bias toward unsafe following**
        unsafe_offset = random.uniform(-2.0, -1.0)  # makes distance too close
        distance_error = distance_to_leader - (desired_distance + unsafe_offset)

        Kp_acc = 0.5
        acc_cmd = Kp_acc * distance_error
        acc_cmd = np.clip(acc_cmd, 1.0, 4.0)  # clipped to always push forward

        # === Optional destabilization when aligned too well ===
        if distance_to_leader < 4.0 and abs(heading_error_rad) < 0.1:
            acc_cmd += 1.0  # forces overshoot at close range

        # === Random noise injection (~30% chance) ===
        if random.random() < 0.3:
            acc_cmd += np.random.normal(0.5, 1.5)
            steer_cmd += np.random.normal(0.0, 0.2)
            acc_cmd = np.clip(acc_cmd, 1.5, 4.0)
            steer_cmd = np.clip(steer_cmd, -1.0, 1.0)

        return np.array([acc_cmd, steer_cmd], dtype=np.float32)

