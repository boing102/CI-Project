import os
import numpy as np
from pynput.keyboard import Listener, KeyCode
from pytocl.car import State, Command
from pytocl.driver import Driver

_dir = os.path.dirname(os.path.realpath(__file__))

# Whether WASD keys are currently pressed.
accelerate = False
brake = False
left = False
right = False

# Each element is a state. Like the training data.
collected_data = []


# Switch a WASD key to pressed state.
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


# Switch a WASD key to released state.
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

    # Start listening for WASD key actions on second thread.
    def listener(self):
        print("Started listener")
        return Listener(on_press=on_press, on_release=on_release)

    # Steer using the inherited steering control method.
    def steer(self, command, carstate):
        if right:
            command.steering = self.steering_ctrl.control(-1, carstate.current_lap_time) 
        elif left:
            command.steering = self.steering_ctrl.control(1, carstate.current_lap_time) 

    # Set acceleration, brake and gear.
    def acc_brake(self, command, carstate):
        if accelerate:
            target = 100 if carstate.speed_x < 20 else carstate.speed_x ** 1.5
            self.accelerate(carstate, target, command)
            if command.gear < 1:
                command.gear = 1
            if carstate.speed_x < 0:
                command.brake = 1
        if brake:
            if carstate.speed_x < 0.1:
                command.gear = -1
                command.accelerator = 1
            else:
                command.brake = 1

    # Given the car State return the next Command.
    def drive(self, carstate: State) -> Command:
        command = Command()
        self.steer(command, carstate)
        self.acc_brake(command, carstate)
        return command
