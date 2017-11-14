import logging
from keras.models import load_model
import os
from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
from keras_0 import PCAFunction

_logger = logging.getLogger(__name__)
_dir = os.path.dirname(os.path.realpath(__file__))

"""
Definitions of State and Command:

State: https://github.com/moltob/pytocl/blob/master/pytocl/car.py#L28
Command: https://github.com/moltob/pytocl/blob/master/pytocl/car.py#L108
"""


# Given a State return a list of sensors for our NN.
def sensor_list(carstate):
    return np.concatenate([
        [carstate.speed_x],#2
        [carstate.race_position], #1
        [carstate.angle],
        carstate.distances_from_edge, #3
        # carstate.current_lap_time,
        # carstate.damage,
        # carstate.distance_from_start: Best possible distance 
        # carstate.distance_raced, 1: How good or bad your driving
        # carstate.fuel,
        # carstate.gear,
        # carstate.last_lap_time,
        # carstate.opponents,
        # carstate.rpm,
        # carstate.speed_y,2 
        # carstate.speed_z,2
        # carstate.distance_from_center,
        # carstate.wheel_velocities,
        # carstate.z,
        # carstate.focused_distances_from_edge 3
    ]).reshape(1, 22)

class MyDriver(Driver):

    def __init__(self, *args, **kwargs):
        self.nn = load_model(os.path.join(_dir, "./models/keras.pickle"))
        super(MyDriver, self).__init__(*args, **kwargs)

    def drive(self, carstate: State) -> Command:
        """Given the car State return the next Command.

        Command attributes:

        accelerator: Accelerator, 0: no gas, 1: full gas, [0;1].
        brake:  Brake pedal, [0;1].
        gear: Next gear. -1: reverse, 0: neutral, [1;6]: corresponding forward gear.
        steering: Rotation of steering wheel, -1: full right, 0: straight, 1: full left, [-1;1].
            Full turn results in an approximate wheel rotation of 21 degrees.
        focus: Direction of driver's focus, resulting in corresponding
            ``State.focused_distances_from_edge``, [-90;90], deg.

        """
        x = sensor_list(carstate)

        accelerator, brake, steering = self.nn.predict(x)[0]

        command = Command()
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1

        command.accelerator = accelerator
        command.brake = brake
        command.steering = steering

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command


if __name__ == "__main__":
    nn = load_model(os.path.join(_dir, "./models/keras.pickle"))
    input_ = np.zeros((22, )).reshape(1, 22)
    print(input_.shape)
    out = nn.predict(input_)[0]
    print(out)
