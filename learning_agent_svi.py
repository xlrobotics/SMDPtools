import numpy as np
from gridworld import Tile, Grid_World
# from q_learner import Q_learning
from svi_learner import GridRL as SVI
import pygame, sys, time, random
from pygame.locals import *

import matplotlib.pyplot as plt
# from matplotlib.axes import Axes as ax

def get_features(pos):
    return pos[0]*(board_size[0]) + pos[1] # -1

def decode_features(features):
    return [features/board_size[0], features - board_size[0]*(features/board_size[0]) ]


if __name__ == "__main__":

    threshold = 0.00000000001
    board_size = [13, 13]
    wall = []

    pygame.init()
    #
    # # Set window size and title, and frame delay
    surfaceSize = (13 * 50, 13 * 50)
    #
    # # Create the window
    surface = pygame.display.set_mode(surfaceSize, 0, 0)


    n = board_size[0]*board_size[1]
    board = Grid_World(surface, board_size, wall)

    # setup action based SVI agent
    agent = SVI(alpha=0.5, gamma=0.9, lmbda=0.0, epsilon=0.1, n=n, num_actions=4, room=4)
    agent.set_S()
    # agent.set_P()
    agent.set_goal(board.goal_coord)
    agent.set_WallCord(board.wall_coords)
    agent.set_P()

    # agent.init_V()
    # agent.SVI(threshold)
    # agent.plot("SVI")

    # get agent room information
    blocks = agent.NB_initiation()

    # print agent.room
    # print blocks["R1"]

    # print agent.R

    room_agent = {}
    room_path = {}

    room_path = agent.get_RoomPath()
    # print room_path

    # encapsulate room information into room object, also known as option for higher level agent
    for key in room_path.keys():
        # print key
        room_agent[key] = SVI(alpha=0.5, gamma=0.9, lmbda=0.0, epsilon=0.1, n=n, num_actions=4, room=1)
        room_agent[key].set_S(blocks[key[0]]["inner"]+blocks[key[0]]["wall_cords"])

        room_agent[key].set_goal(list(room_path[key]))
        room_agent[key].set_WallCord(blocks[key[0]]["wall_cords"])
        room_agent[key].set_P()

        room_agent[key].init_V()
        # room_agent[key].init_Policy()
        # room_agent[key].plot("room"+str(key[0]))
        room_agent[key].SVI(threshold)

        room_agent[key].transition_matrix()

        # room_agent[key].plot(key[0]+"to"+key[1])

        agent.add_Opt(key, room_agent[key])

    # run option_svi
    agent.init_V()
    agent.init_Policy()
    agent.SVI(threshold, "Option")

    agent.init_V()
    agent.init_Policy()
    agent.Ifplot = True
    agent.SVI(threshold)
    # agent.init_V()


    agent.plot_curve(agent.action_diff, agent.hybrid_diff,  "compare diff")
    agent.plot_curve(agent.action_val, agent.hybrid_val, "compare val")
    agent.plot_curve(agent.action_pi_diff, agent.hybrid_pi_diff, "compare pi")

    # room_agent[("R1", "R2")].plot("R1_to_R2_SVI")
    # print room_agent[("R1", "R2")].Policy







