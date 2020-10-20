import pygame
import numpy as np
import copy
import time

pygame.init()
Window_size = 400
block_size = 20
gameDisplay = pygame.display.set_mode((Window_size,Window_size))
gameDisplay.fill((0, 255, 0))
pygame.time.delay(50)

print("Welcome to Snake")
game_on = True
colors = {
            "Grass": (0,255,0),
            "Tail_one": (0,0,255),
            "Snake_one": (0,0,255),
            "Head_one": (0,0,255),
            "Tail_two": (0,0,0),
            "Snake_two": (0,0,0),
            "Head_two": (0,0,0),
            "Food": (255,0,0)
}
board = {}
for i in range(block_size):
    for j in range(block_size):
        board[(i,j)] = ["Grass",[0,0]]

Snake_one = [[7,7],[6,7],[5,7]]
Snake_two = [[13,13],[14,13],[15,13]]
foods = []
board[(5,7)] = ["Tail_one",[1,0]]
board[(6,7)] = ["Snake_one",[1,0]]
board[(7,7)] = ["Head_one",[1,0]]

board[(15,13)] = ["Tail_two",[-1,0]]
board[(14,13)] = ["Snake_two",[-1,0]]
board[(13,13)] = ["Head_two",[-1,0]]


rate = 5000
max_food = 5
food_appeared = 0


def draw_grid():

    grid_points = np.linspace(0,Window_size-block_size,int(Window_size/block_size))
    #print(grid_points)
    for i in range(len(grid_points)):
        for j in range(len(grid_points)):
            pygame.draw.rect(gameDisplay,colors[board[i,j][0]],(grid_points[i],grid_points[j],block_size,block_size),0)


while game_on:

    p_one = None
    keys=pygame.key.get_pressed()
    if keys[pygame.K_DOWN]:
        p_one = [0,1]
    if keys[pygame.K_UP]:
        p_one = [0,-1]
    if keys[pygame.K_LEFT]:
        p_one = [-1,0]
    if keys[pygame.K_RIGHT]:
        p_one = [1,0]

    p_two = None
    if keys[pygame.K_s]:
        p_two = [0,1]
    if keys[pygame.K_w]:
        p_two = [0,-1]
    if keys[pygame.K_a]:
        p_two = [-1,0]
    if keys[pygame.K_d]:
        p_two = [1,0]

    pygame.time.delay(100)

    ##### Snake_one
    grow_snake = False
    tail_reset = False
    tail_momentum = [0,0]
    prev_tail = copy.deepcopy(Snake_one[-1])
    for index,segment in enumerate(Snake_one):
        plc = tuple(segment)
        if index == 0:
            if p_one==None or (board[plc][1][0]==-p_one[0]  and board[plc][1][1]==-p_one[1]):
                p_one = board[plc][1]
            else:
                board[plc][1] = copy.deepcopy(p_one)

            if (board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] == 'Snake_one' or
                board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] == 'Snake_two' or
                board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] == 'Head_two'):
                game_on = False
            if (board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] == 'Food'):
                grow_snake = True
            if (board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] == 'Tail_one'):
                tail_reset = True
                tail_momentum = copy.deepcopy(board[(segment[0]+p_one[0],segment[1]+p_one[1])][1])
            board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] = 'Head_one'
            board[(segment[0]+p_one[0],segment[1]+p_one[1])][1] = copy.deepcopy(p_one)

        elif index != len(Snake_one)-1:
            p_one = board[plc][1]
            board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] = 'Snake_one'
        else:
            if grow_snake == False:
                if tail_reset == False:
                    p_one = copy.deepcopy(board[plc][1])
                    board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] = 'Tail_one'
                    board[plc][1] = [0,0]
                    board[plc][0] = 'Grass'
                else:
                    p_one = tail_momentum
                    board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] = "Tail_one"


            else:
                p_one = board[plc][1]
                board[(segment[0]+p_one[0],segment[1]+p_one[1])][0] = 'Snake_one'
                board[plc][0] = 'Tail_one'
        Snake_one[index] = [segment[0]+p_one[0],segment[1]+p_one[1]]
    if grow_snake:
        Snake_one.append(prev_tail)


    ##### Snake Two
    grow_snake = False
    tail_reset = False
    tail_momentum = [0,0]
    prev_tail = copy.deepcopy(Snake_two[-1])
    for index,segment in enumerate(Snake_two):
        plc = tuple(segment)
        if index == 0:
            if p_two==None or (board[plc][1][0]==-p_two[0]  and board[plc][1][1]==-p_two[1]):
                p_two = board[plc][1]
            else:
                board[plc][1] = copy.deepcopy(p_two)

            if (board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] == 'Snake_two' or
                board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] == 'Head_one' or
                board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] == 'Snake_one' or
                board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] == 'Tail_one'):
                game_on = False
            if (board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] == 'Food'):
                grow_snake = True
            if (board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] == 'Tail_two'):
                tail_reset = True
                tail_momentum = copy.deepcopy(board[(segment[0]+p_two[0],segment[1]+p_two[1])][1])
            board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] = 'Head_two'
            board[(segment[0]+p_two[0],segment[1]+p_two[1])][1] = copy.deepcopy(p_two)

        elif index != len(Snake_two)-1:
            p_two = board[plc][1]
            board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] = 'Snake_two'
        else:
            if grow_snake == False:
                if tail_reset == False:
                    p_two = copy.deepcopy(board[plc][1])
                    board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] = 'Tail_two'
                    board[plc][1] = [0,0]
                    board[plc][0] = 'Grass'
                else:
                    p_two = tail_momentum
                    board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] = "Tail_two"
            else:
                p_two = board[plc][1]
                board[(segment[0]+p_two[0],segment[1]+p_two[1])][0] = 'Snake_two'
                board[plc][0] = 'Tail_two'
        #print(p_two)
        Snake_two[index] = [segment[0]+p_two[0],segment[1]+p_two[1]]
    if grow_snake:
        Snake_two.append(prev_tail)



    if (pygame.time.get_ticks() > (food_appeared +1)*rate and len(foods) < max_food):
        x_pos = np.random.randint(0,block_size)
        y_pos = np.random.randint(0,block_size)
        while board[(x_pos,y_pos)][0] != 'Grass':
            x_pos = np.random.randint(0,block_size)
            y_pos = np.random.randint(0,block_size)
        board[(x_pos,y_pos)][0] = 'Food'
        food_appeared += 1
    draw_grid()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_on = False

pygame.quit()
