import numpy as np
import redis
import time
from enum import Enum
import json

redis_client = redis.Redis()

class RedisKeys(Enum):
    CURRENT_CARTESIAN_POSITION = "sim::fr3::current_cartesian_position"
    CURRENT_CARTESIAN_ORIENTATION = "sim::fr3::current_cartesian_orientation"
    DESIRED_CARTESIAN_POSITION = "sim::fr3::desired_cartesian_position"
    DESIRED_CARTESIAN_ORIENTATION = "sim::fr3::desired_cartesian_orientation"
    RESET = "sim::fr3::reset"

def main():

    # print("Starting the control simulation on the python end...")

    #High level Controller with access to the keys that we are interested in ---- this is good for now ----

    #Needed for intermediate testing of the position control
    pass

def moveToPos(desired_pos: np.array, desired_orient = np.array([[1,0,0], [0, -1, 0], [0, 0, -1]]), max_iters = 200):

    redis_client.set(RedisKeys.DESIRED_CARTESIAN_POSITION.value, json.dumps(desired_pos.tolist()))
    redis_client.set(RedisKeys.DESIRED_CARTESIAN_ORIENTATION.value, json.dumps(desired_orient.tolist()))

    for i in range(max_iters):
        curr_pos = getCurrentPosition()

        if np.linalg.norm(curr_pos - desired_pos) < 1e-2:
            return
        
        time.sleep(0.01)

def setCurrentPosition(desired_pos: np.array, desired_orient = np.array([[1,0,0], [0, -1, 0], [0, 0, -1]])):
    redis_client.set(RedisKeys.CURRENT_CARTESIAN_POSITION.value, json.dumps(desired_pos.tolist()))
    redis_client.set(RedisKeys.CURRENT_CARTESIAN_ORIENTATION.value, json.dumps(desired_orient.tolist()))

def getCurrentPosition() -> np.array:
    return np.array(json.loads(redis_client.get(RedisKeys.CURRENT_CARTESIAN_POSITION.value)))

if __name__ == "__main__":
    main()