import numpy as np
import pygame
import time
import os

# load
with open('test_game.npy', 'rb') as f:
    game_history = np.load(f,allow_pickle=True)

pygame.init()

field_width = 37.
endzone_length = 18.
field_length = endzone_length * 2 + 64
num_players_per_team = 7
attributes_per_player = 3
player_circle_radius = 1
cone_circle_radius = 1

time_per_state = 1 # seconds
scale = 3
width = field_width
height = field_length
green = '0x023d03'

screen = pygame.display.set_mode((scale*width,scale*height))

done = False
time.sleep(2)
print('playing game of %d steps' %game_history.shape[0])
#print(game_history[3])

# check if someone has frisbee in every step
def check_frisbee_disappears():
    state_value = 0
    for state in game_history:
        frisbee_in_state = False
        for i in range(0,num_players_per_team*attributes_per_player,attributes_per_player): # offense
            player_has_frisbee = state[i+2]
            if player_has_frisbee:
                frisbee_in_state = True
        if not frisbee_in_state:
            print('frisbee not in state %d' %state_value)
            return
        state_value += 1
    
check_frisbee_disappears()
for state in game_history:
    screen.fill(green)
    # boundaries
    pygame.draw.circle(screen,pygame.Color('0xe8a54d'), [0, 0], cone_circle_radius*scale)
    pygame.draw.circle(screen,pygame.Color('0xe8a54d'), [field_width*scale, 0], cone_circle_radius*scale)
    pygame.draw.circle(screen,pygame.Color('0xe8a54d'), [field_width*scale, endzone_length*scale], cone_circle_radius*scale)
    pygame.draw.circle(screen,pygame.Color('0xe8a54d'), [0, endzone_length*scale], cone_circle_radius*scale)
    pygame.draw.circle(screen,pygame.Color('0xe8a54d'), [0, field_length*scale], cone_circle_radius*scale)
    pygame.draw.circle(screen,pygame.Color('0xe8a54d'), [field_width*scale, field_length*scale], cone_circle_radius*scale)
    pygame.draw.circle(screen,pygame.Color('0xe8a54d'), [field_width*scale, (field_length-endzone_length)*scale], cone_circle_radius*scale)
    pygame.draw.circle(screen,pygame.Color('0xe8a54d'), [0, (field_length-endzone_length)*scale], cone_circle_radius*scale)

    for i in range(0,num_players_per_team*attributes_per_player,attributes_per_player): # offense
        player_x = state[i]
        player_y = state[i+1]
        player_has_frisbee = state[i+2]
        if player_has_frisbee == 1.:
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