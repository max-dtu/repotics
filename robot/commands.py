"""
Command vocabulary for the robot.

All robot motion and actuator commands are defined here as a single enum so
that every other module (Commander, Agent, Planner) imports from one place.
Adding a new command is a one-line change in this file.
"""
from enum import Enum


class Command(str, Enum):
    """
    Discrete commands the robot understands.

    Because Command inherits from str, instances are directly serialisable
    (e.g. JSON, logging, serial framing) without any extra conversion.

        >>> Command.FORWARD          # <Command.FORWARD: 'forward'>
        >>> str(Command.FORWARD)     # 'forward'
        >>> Command("left")          # <Command.LEFT: 'left'>
    """

    FORWARD       = "forward"
    BACKWARD      = "backward"
    LEFT          = "left"
    RIGHT         = "right"
    OPEN_GRIPPER  = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"


#: Flat list used by the random policy and the REPL banner.
AVAILABLE_COMMANDS: list[Command] = list(Command)
