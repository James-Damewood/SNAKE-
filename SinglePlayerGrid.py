import pygame
import numpy as np
import copy

pygame.init()
Window_size = 400
block_size = 20
gameDisplay = pygame.display.set_mode((Window_size,Window_size))
gameDisplay.fill((255, 255, 255))
print("Welcome to Snake")
game_on = True
colors = {
            "Grass": (0,255,0),
            "Tail": (0,0,255),
            "Snake": (0,0,255),
            "Head": (0,0,255),
            "Food": (255,0,0)
}
board = {}
for i in range(block_size):
    for j in range(block_size):
        board[(i,j)] = ["Grass",[0,0]]

Snake = [[7,10],[6,10],[5,10]]
foods = []
board[(5,10)] = ["Tail",[1,0]]
board[(6,10)] = ["Snake",[1,0]]
board[(7,10)] = ["Head",[1,0]]
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

    p = None
    keys=pygame.key.get_pressed()
    if keys[pygame.K_DOWN]:
        p = [0,1]
    if keys[pygame.K_UP]:
        p = [0,-1]
    if keys[pygame.K_LEFT]:
        p = [-1,0]
    if keys[pygame.K_RIGHT]:
        p = [1,0]

    pygame.time.delay(50)
    grow_snake = False
    prev_tail = copy.deepcopy(Snake[-1])
    for index,segment in enumerate(Snake):
        plc = tuple(segment)
        if index == 0:
            if p==None or (board[plc][1][0]==-p[0]  and board[plc][1][1]==-p[1]):
                p = board[plc][1]
            else:
                board[plc][1] = copy.deepcopy(p)

            if (board[(segment[0]+p[0],segment[1]+p[1])][0] == 'Snake'):
                game_on = False
            if (board[(segment[0]+p[0],segment[1]+p[1])][0] == 'Food'):
                grow_snake = True
            board[(segment[0]+p[0],segment[1]+p[1])][0] = 'Head'
            board[(segment[0]+p[0],segment[1]+p[1])][1] = copy.deepcopy(p)

        elif index != len(Snake)-1:
            p = board[plc][1]
            board[(segment[0]+p[0],segment[1]+p[1])][0] = 'Snake'
        else:
            if grow_snake == False:
                p = board[plc][1]
                board[(segment[0]+p[0],segment[1]+p[1])][0] = 'Tail'
                board[plc][1] = [0,0]
                board[plc][0] = 'Grass'
            else:
                p = board[plc][1]
                board[(segment[0]+p[0],segment[1]+p[1])][0] = 'Snake'
                board[plc][0] = 'Tail'
        Snake[index] = [segment[0]+p[0],segment[1]+p[1]]
    if grow_snake:
        Snake.append(prev_tail)

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
