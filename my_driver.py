import logging
from keras.models import load_model
import os
from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np

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
        [carstate.speed_x],
        [carstate.race_position],
        [carstate.angle],
        carstate.distances_from_edge,
        # carstate.current_lap_time,
        # carstate.damage,
        # carstate.distance_from_start,
        # carstate.distance_raced,
        # carstate.fuel,
        # carstate.gear,
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


class MyDriver(Driver):

    def __init__(self, *args, **kwargs):
        self.nn = load_model(os.path.join(_dir, "./models/keras.pickle"))
        super(MyDriver, self).__init__(*args, **kwargs)

    def drive(self, carstate: State) -> Command:
        """Produces driving command in response to newly received car state. This is a
        dummy driving routine, very dumb and not really considering a lot of
        inputs. But it will get the car (if not disturbed by other drivers)
        successfully driven along the race track.

        """
        x = sensor_list(carstate)
        print(len(x))
        # print(self.nn(x))

        command = Command()
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
        self.steer(carstate, 0.0, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command


if __name__ == "__main__":
    nn = load_model(os.path.join(_dir, "./models/keras.pickle"))
    input_ = np.zeros((22, )).reshape(1, 22)
    out = nn.predict(input_)[0]
    print(out)
