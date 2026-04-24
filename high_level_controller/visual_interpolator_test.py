#Lets generate 3 action chunks, with random quaternions

#one cool test that we can do is the following --- tell the robot to go up, and then have a chunk
# to tell it to go to the left --- I think this would be really good --- 

import time
import numpy as np
from interpolator import TrajectoryInterpolator
import redis
from scipy.spatial.transform import Rotation as R

DESIRED_CARTESIAN_POSITION = "sai::sim::franka::desired_cartesian_position"
DESIRED_CARTESIAN_ORIENTATION = "sai::sim::franka::desired_cartesian_orientation"

action_hz = 10
n_actions = 4
chunk_size = 8

redis_client = redis.Redis()

def generate_straight_action_chunk(p1, p2): 
    orientation = np.array([[1,0,0], [0, -1, 0], [0,0,-1]])
    quat = R.from_matrix(orientation).as_quat()

    pos_chunk = np.linspace(p1, p2, chunk_size)
    quat_stacked = np.broadcast_to(quat, (chunk_size , 4 ))
    action_chunk = np.hstack((pos_chunk, quat_stacked))

    return action_chunk


chunk1 = generate_straight_action_chunk(np.array([0.4, 0.0, 0.36]), np.array([0.4,0.0,0.45]))  
chunk2 = generate_straight_action_chunk(np.array([0.4,0.0,0.45]), np.array([0.4,0.2, 0.45])) 

chunk_list = [chunk1, chunk2]

def schedule_action_chunks(action_chunks):

    action_chunk_index = -1
    inference_sent = False
    interpolator = TrajectoryInterpolator(redis_client, desired_position_key=DESIRED_CARTESIAN_POSITION, desired_orientation_key= DESIRED_CARTESIAN_ORIENTATION)
    interpolator.start()


    while action_chunk_index < len(action_chunks):
        if not inference_sent:
            action_chunk_index += 1
            inference_sent_time = time.time()
            interpolator.enqueue_chunk(action_chunks[action_chunk_index], ts = np.arange(inference_sent_time, inference_sent_time + action_chunks[action_chunk_index].shape[0] *(1/action_hz), 1/action_hz))
            inference_sent = True
            

        if inference_sent and time.time() > inference_sent_time + n_actions * (1/action_hz):
            inference_sent = False

        time.sleep(0.001)

print("Starting to run the action chunks: ")

time.sleep(5)

schedule_action_chunks(chunk_list)

#The mid level interpolator is moving correctly --- which is really good --











