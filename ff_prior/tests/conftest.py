import pytest
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.initial_state import InitialState
from ff_prior.geometry import CollisionBackend


@pytest.fixture(scope="session")
def physical():
    robot = RobotAdapter()
    initial = InitialState.capture(robot,robot.new_data())
    return robot,initial,CollisionBackend(robot)
