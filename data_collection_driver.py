import os
import numpy as np
from pynput.keyboard import Listener, KeyCode
from pytocl.car import State, Command
from pytocl.driver import Driver

_dir = os.path.dirname(os.path.realpath(__file__))

# Whether keys are currently pressed.
accelerate = False
brake = False
left = False
right = False


def on_press(key):
    if key == KeyCode.from_char("w"):
        global accelerate
        accelerate = True
    if key == KeyCode.from_char("s"):
        global brake
        brake = True
    if key == KeyCode.from_char("a"):
        global left
        left = True
    if key == KeyCode.from_char("d"):
        global right
        right = True


def on_release(key):
    if key == KeyCode.from_char("w"):
        global accelerate
        accelerate = False
    if key == KeyCode.from_char("s"):
        global brake
        brake = False
    if key == KeyCode.from_char("a"):
        global left
        left = False
    if key == KeyCode.from_char("d"):
        global right
        right = False


class DataCollectionDriver(Driver):

    def listener(self):
        print("Started listener")
        return Listener(on_press=on_press, on_release=on_release)

    def steer(self, current_lap_time):
        if right:
            return self.steering_ctrl.control(-1, current_lap_time) 
        if left:
            return self.steering_ctrl.control(1, current_lap_time) 

    def acc_brake(self, command, carstate):
        if accelerate:
            self.accelerate(carstate, carstate.speed_x + 500, command)
        if accelerate and command.gear < 1:
            command.gear = 1
        command.brake = int(brake)

    # Given the car State return the next Command.
    def drive(self, carstate: State) -> Command:
        command = Command()
        self.acc_brake(command, carstate)
        command.steering = self.steer(carstate.current_lap_time)
        return command
