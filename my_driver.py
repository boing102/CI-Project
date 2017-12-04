import logging
import os
import sys
import time
from pytocl.driver import Driver
from pytocl.car import State, Command
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import numpy as np
import pickle

_logger = logging.getLogger(__name__)
_dir = os.path.dirname(os.path.realpath(__file__))
path_to_model = os.path.join(_dir, "models/sklearn.pickle")
path_to_pca = os.path.join(_dir, "models/pca.pickle")

"""
Definitions of State and Command:
State: https://github.com/moltob/pytocl/blob/master/pytocl/car.py#L28
Command: https://github.com/moltob/pytocl/blob/master/pytocl/car.py#L108

State attributes:
    sensor_dict: Dictionary of sensor key value pairs in string form.
    angle: Angle between car direction and track axis, [-180;180], deg.
    current_lap_time: Time spent in current lap, [0;inf[, s.
    damage: Damage points, 0 means no damage, [0;inf[, points.
    distance_from_start: Distance of car from start line along track center, [0;inf[, m.
    distance_raced: Distance car traveled since beginning of race, [0;inf[, m.
    fuel: Current fuel level, [0;inf[, l.
    gear: Current gear. -1: reverse, 0: neutral, [1;6]: corresponding forward gear.
    last_lap_time: Time it took to complete last lap, [0;inf[, s.
    opponents: Distances to nearest opponents in 10 deg slices in [-180;180] deg. [0;200], m.
    race_position: Position in race with respect to other cars, [1;N].
    rpm: Engine's revolutions per minute, [0;inf[.
    speed_x: Speed in X (forward) direction, ]-inf;inf[, m/s.
    speed_y: Speed in Y (left) direction, ]-inf;inf[, m/s.
    speed_z: Speed in Z (up) direction, ]-inf;inf[, m/s.
    distances_from_edge: Distances to track edge along configured driver range finders,
        [0;200], m.
    focused_distances_from_edge: Distances to track edge, five values in five degree range along
        driver focus, [0;200], m. Can be used only once per second and while on track,
        otherwise values set to -1. See ``focused_distances_from_egde_valid``.
    distance_from_center: Normalized distance from track center, -1: right edge, 0: center,
        1: left edge, [0;1].
    wheel_velocities: Four wheels' velocity, [0;inf[, deg/s.
    z: Distance of car center of mass to track surface, ]-inf;inf[, m.

Command attributes:
    accelerator: Accelerator, 0: no gas, 1: full gas, [0;1].
    brake:  Brake pedal, [0;1].
    gear: Next gear. -1: reverse, 0: neutral, [1;6]: corresponding forward gear.
    steering: Rotation of steering wheel, -1: full right, 0: straight, 1: full left, [-1;1].
        Full turn results in an approximate wheel rotation of 21 degrees.
    focus: Direction of driver's focus, resulting in corresponding
        ``State.focused_distances_from_edge``, [-90;90], deg.
"""

# Path of carstate given car ID.
carstate_filepath = lambda car_id: os.path.join(_dir, "carstate-{0}".format(car_id))


# Save carstate to file.
def save_carstate(car_id, carstate):
    with open(carstate_filepath(car_id), "wb") as f:
        pickle.dump(carstate, f)


# Load carstate given car ID.
# May return None on rare occasion.
def load_carstate(car_id):
    try:
        with open(carstate_filepath(car_id), "rb") as f:
            return pickle.load(f)
    except:
        return None

# Adds random ID to a car ID file and return ID.
def set_car_id():
    car_id = np.random.randint(sys.maxsize)
    with open(os.path.join(_dir, "car_ids"), "a") as f:
        f.write("\n{0}".format(car_id))
    return car_id


# Get other car ID from list of IDs in file.
def get_other_car_id(car_id):
    with open(os.path.join(_dir, "car_ids")) as f:
        ids = f.readlines()[-2:]  # Most recent 2.
    ids = list(filter(lambda x: int(x) != car_id, ids))
    return int(ids[0].strip())


class MyDriver(Driver):

    def __init__(self, *args, **kwargs):
        self.car_id = set_car_id()
        with open(path_to_model, 'rb') as handle:
            self.nn = pickle.load(handle)
        with open(path_to_pca, 'rb') as handle:
            self.pca = pickle.load(handle)

        super(MyDriver, self).__init__(*args, **kwargs)
        self.reset_counter = 0
        self.reverse_counter = 0
        self.reverse_start = False
        self.old_distance = 0
        self.reverseCondition = False
        self.nn_counter = 0
        self.speed = 0

    # Given the car State return the next Command.
    def drive(self, carstate: State) -> Command:
<<<<<<< HEAD
=======
        other_car_id = get_other_car_id(self.car_id)
        other_carstate = load_carstate(other_car_id)

>>>>>>> a4d24984dfe5a5b1876fcfb34ba56d3f9d3a703f
        command = Command()
        #Get data
        x_new = self.sensor_list(carstate)        
        x_new_norm = normalize(x_new)
        x_new_norm = self.pca.transform(x_new_norm)

        #Predict and use predictions
        prediction = self.nn.predict(x_new_norm)[0]        
        command.accelerator = prediction[0]
        command.brake = prediction[1] #To test for correction
        steering = prediction[2]

        #If we drive fast, don't steer sharply
        if self.speed < 100:
            steering = steering
            if np.absolute(steering) > 0.25:
                command.accelerator = command.accelerator * 0.3
        elif self.speed > 100 and self.speed < 130:
            steering = steering * 0.7
        else:
            steering = steering * 0.5

        # if command.brake > 0.3:
        #     steering = 0

        command.steering = steering


        # Gear is set by a deterministic rule.
        if carstate.rpm > 9000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 4500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1

        # command.steering = prediction[2]

        #Hardcode when we are stuck,not at the start
        self.reset_counter += 1
        if self.reverse_start:
            self.nn_counter +=1
            #Handle gears
            command.gear = 1 
            command.brake = 0
            command.accelerator = 0.33
            command.steering = (carstate.angle - 2*carstate.distance_from_center)/(180/21)
            #Brake first
            if self.nn_counter < 50:
                command.brake = 1
                command.accelerator = 0
            #After certain moment, let NN take over again:
            if (self.nn_counter > 600 or self.speed > 35) and carstate.distance_from_center <0.1:
                self.reverse_start = False
                self.nn_counter = 0

        #Drive backwards to correct
        if (self.speed < 1 and self.reset_counter> 100 and not self.reverse_start) or self.reverseCondition:
            self.reverse_counter += 1
            self.reverseCondition = True
            #Handle gears
            a = carstate.angle
            command.steering = (-a)/(180/21)
            command.gear = -1
            command.accelerator = 0.5
            command.brake = 0
            if self.reverse_counter >100:
                  self.reverse_counter = 0
                  self.reverse_start = True
                  self.reverseCondition = False
                  command.brake = 1

<<<<<<< HEAD
        #When we are moving into the wrong direction. Exception on start/finish
        if (carstate.distance_from_start < self.old_distance) and carstate.distance_from_start >10:
             self.reverse_start = True

        #Update distance
        self.old_distance = carstate.distance_from_start
        print(self.reverseCondition, self.reverse_start,self.speed,command.steering, carstate.angle, carstate.distance_from_center)
 # We don't set driver focus, or use focus edges.
=======
        # Hardcode when we are moving into the wrongdirection, assume nose backwards
        if (False and carstate.distance_from_start < self.old_distance):
             self.reverseCondition = True
             print("Wrong way, yo")
        # Update distance
        self.old_distance = carstate.distance_from_start
        # We don't set driver focus, or use focus edges.
>>>>>>> a4d24984dfe5a5b1876fcfb34ba56d3f9d3a703f
        if self.data_logger:
            self.data_logger.log(carstate, command)

        save_carstate(self.car_id, carstate)
        return command

    # Given a State return a list of sensors for our NN.
    def sensor_list(self, carstate):
        # Speed from the three velocities x, y, z.
        self.speed = np.sqrt(np.sum([s**2 for s in (carstate.speed_x, carstate.speed_y,
                                               carstate.speed_z)])) * (18/5)
        return np.concatenate([
            [self.speed],
            [carstate.distance_from_center],
            [carstate.angle],
            carstate.distances_from_edge,
            # carstate.current_lap_time,
            # carstate.damage,
            # carstate.distance_from_start,
            # carstate.distance_raced,
            # carstate.fuel,
            #[carstate.gear],
            # carstate.last_lap_time,
            # carstate.opponents,
            # carstate.rpm,
            # carstate.speed_y,
            # carstate.speed_z,
            # carstate.distance_from_center,
            # carstate.wheel_velocities,
            # carstate.z,
            # carstate.focused_distances_from_edge
        ]).reshape(1, 22)

if __name__ == "__main__":
    nn = load_model(os.path.join(_dir, "./models/sklearn.pickle"))
    input_ = np.zeros((22, )).reshape(1, 22)
    print(input_.shape)
    out = nn.predict(input_)[0]
    print(out)
