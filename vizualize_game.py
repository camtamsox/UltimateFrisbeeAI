import numpy as np
import pygame
import time
import os

# load
with open('test_game.npy', 'rb') as f:
    game_history = np.load(f)

pygame.init()

field_width = 37.
endzone_length = 18.
field_length = endzone_length * 2 + 64
num_players_per_team = 7
attributes_per_player = 3
player_circle_radius = 1

time_per_state = 0.15 # seconds
scale = 3
width = field_width
height = field_length
green = '0x023d03'

screen = pygame.display.set_mode((scale*width,scale*height))

done = False
time.sleep(2)
print(game_history.shape)
for state in game_history:
    screen.fill(green)
    for i in range(0,num_players_per_team*attributes_per_player,attributes_per_player): # offense
        player_x = state[i]
        player_y = state[i+1]
        player_has_frisbee = state[i+2]
        if player_has_frisbee:
            pygame.draw.circle(screen,pygame.Color('0xffffff'), [round(scale*player_x), round(scale*player_y)], scale*player_circle_radius*2)
        pygame.draw.circle(screen,pygame.Color('0x032e73'), [round(scale*player_x), round(scale*player_y)], scale*player_circle_radius)
    for i in range(num_players_per_team*attributes_per_player,2*num_players_per_team*attributes_per_player,attributes_per_player): # defense
        player_x = state[i]
        player_y = state[i+1]
        player_has_frisbee = state[i+2]
        pygame.draw.circle(screen,pygame.Color('0xde4040'), [round(scale*player_x), round(scale*player_y)], scale*player_circle_radius)

    pygame.display.flip()
    time.sleep(time_per_state)
os.sys.exit()