import csv
import datetime as dt
import os

import numpy as np
from pynput.keyboard import KeyCode, Listener

from my_driver import sensor_list
from pytocl.car import Command, State
from pytocl.driver import Driver

_dir = os.path.dirname(os.path.realpath(__file__))
use_recovery = False


# Whether WASD keys are currently pressed.
accelerate = False
brake = False
left = False
right = False

collected_data = []
recording = False
collected_data_path = None
new_data_path = os.path.join(_dir, "new_data")
if not os.path.exists(new_data_path):
    os.makedirs(new_data_path)


# Unique path to save recorded data to.
def get_path():
    return os.path.join(new_data_path, str(dt.datetime.today()) + ".csv")


# Save one row of collected training data.
def collect_data(carstate, c):
    row = sensor_list(carstate).tolist()
    print(row)
    collected_data.append(row[0])


# Writes the collected data to collected_data_path.
def save_collected_data():
    with open(collected_data_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow([","])
        writer.writerows(collected_data)


# Switch a WASD key to pressed state.
def on_press(key):
    if key == KeyCode.from_char("j"):
        global accelerate
        accelerate = True
    if key == KeyCode.from_char("k"):
        global brake
        brake = True
    if key == KeyCode.from_char("d"):
        global left
        left = True
    if key == KeyCode.from_char("f"):
        global right
        right = True
    if key == KeyCode.from_char("r"):
        global recording
        global collected_data
        global collected_data_path
        if not recording:
            collected_data = []
            collected_data_path = get_path()
            print("Recording data")
            print(len(collected_data))
        else:
            save_collected_data()
            print("Saved collected data to: " + collected_data_path)
        recording = not recording


# Switch a WASD key to released state.
def on_release(key):
    if key == KeyCode.from_char("j"):
        global accelerate
        accelerate = False
    if key == KeyCode.from_char("k"):
        global brake
        brake = False
    if key == KeyCode.from_char("d"):
        global left
        left = False
    if key == KeyCode.from_char("f"):
        global right
        right = False


class DataCollectionDriver(Driver):

    def __init__(self, *args, **kwargs):
        self.state = "normal"
        self.slow_counter = 0
        self.reverse_counter = 0
        super(DataCollectionDriver, self).__init__(*args, **kwargs)

    # Start listening for WASD key actions on second thread.
    def listener(self):
        print("Started listener")
        return Listener(on_press=on_press, on_release=on_release)

    def recovery(self, carstate, command):
        print("State {0}".format(self.state))
        print("slow_counter {0}".format(self.slow_counter))
        print("reverse_counter {0}".format(self.reverse_counter))
        if self.state == "normal":
            if carstate.speed_x < 1:
                self.slow_counter += 1
            else:
                self.slow_counter = 0
            if self.slow_counter > 100:
                self.state = "reverse"
                self.reverse_counter = 0
        if self.state == "reverse":
            command.gear = -1
            command.accelerator = 1
            self.reverse_counter += 1
            if self.reverse_counter > 100:
                self.state = "normal"
                self.slow_counter = 0
                command.gear = 1
        print(command)

    # Steer using the inherited steering control method.
    def steer(self, command, carstate):
        if right:
            command.steering = self.steering_ctrl.control(-1, carstate.current_lap_time) 
        elif left:
            command.steering = self.steering_ctrl.control(1, carstate.current_lap_time) 

    # Set acceleration, brake and gear.
    def acc_brake(self, command, carstate):
        if accelerate:
            # A strange target rule, because speed is incorrectly reported?
            target = 100 if carstate.speed_x < 20 else carstate.speed_x ** 1.5
            self.accelerate(carstate, target, command)
            # Come out of neutral.
            if command.gear < 1:
                command.gear = 1
            # To accelerate when reversing, we first need to stop.
            if brake or carstate.speed_x < 0:
                command.brake = 1
        elif brake:
            # To reverse we apply full accelerator in reverse gear.
            if carstate.speed_x < 0.1:
                command.gear = -1
                command.accelerator = 1
            # Else we apply full brake, trying to come to a stop.
            else:
                command.brake = 1

    # Given the car State return the next Command.
    def drive(self, carstate: State) -> Command:
        command = Command()
        self.steer(command, carstate)
        self.acc_brake(command, carstate)
        if recording:
            collect_data(carstate, command)
        if use_recovery:
            self.recovery(carstate, command)
        return command
