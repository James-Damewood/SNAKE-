import numpy as np
import copy
import torch

class Game():
    def __init__(self,model = None):
        self.model = model
        # 0 Grass
        # 1 Tail
        # 2 Snake
        # 3 Head
        # 4 Apple
        self.Window_size = 100
        self.block_size = 10
        self.board = {}
        self.p = {}
        self.board = np.zeros((self.block_size,self.block_size))
        for i in range(self.block_size):
            for j in range(self.block_size):
                self.p[(i,j)] = [0,0]
        self.Snake = [[3,5],[2,5],[1,5]]
        self.board[1,5] = 1
        self.p[(1,5)] = [1,0]
        self.board[2,5] = 2
        self.p[(2,5)] = [1,0]
        self.board[3,5] = 3
        self.p[(3,5)] = [1,0]
        self.foods = []
        self.max_food = 75
        self.food_appeared = 0
        self.rate = 1
        self.num_steps = 0

    def play_game(self):
        alive = True
        length = 3
        self.num_steps = 0
        while alive:
            alive,length = self.game_step()
            self.num_steps += 1
        return length,self.num_steps

    def get_move(self):
        move = self.model.forward(self.board)
        return move

    def game_step(self,action):
        game_on = True
        p_stp = None
        if action == 1:
            p_stp = [1,0]
        elif action == 2:
            p_stp = [-1,0]
        elif action == 3:
            p_stp = [0,1]
        elif action == 4:
            p_stp = [0,-1]
        else:
            p_stp = [0,0]


        grow_snake = False
        prev_tail = copy.deepcopy(self.Snake[-1])
        for index,segment in enumerate(self.Snake):
            plc = tuple(segment)
            if index == 0:
                if plc in self.p:
                    if ((p_stp[0]==0 and p_stp[1]==0) or (self.p[plc][0]==-p_stp[0]  and self.p[plc][1]==-p_stp[1])):
                        p_stp = self.p[plc]
                    else:
                        self.p[plc] = copy.deepcopy(p_stp)
                else:
                    game_on = False
                if (segment[0]+p_stp[0]>=0 and
                    segment[1]+p_stp[1]>=0 and
                    segment[0]+p_stp[0]<10 and
                    segment[1]+p_stp[1]<10  ):
                    if (self.board[segment[0]+p_stp[0],segment[1]+p_stp[1]] == 2):
                        game_on = False
                    if (self.board[segment[0]+p_stp[0],segment[1]+p_stp[1]] == 4):
                        grow_snake = True
                    self.board[segment[0]+p_stp[0],segment[1]+p_stp[1]] = 3
                    self.p[(segment[0]+p_stp[0],segment[1]+p_stp[1])] = copy.deepcopy(p_stp)
                else:
                    game_on = False


            elif index != len(self.Snake)-1:
                if game_on:
                    p_stp = self.p[plc]
                    self.board[segment[0]+p_stp[0],segment[1]+p_stp[1]] = 2
            else:
                if game_on:
                    if grow_snake == False:
                        p_stp = self.p[plc]
                        self.board[segment[0]+p_stp[0],segment[1]+p_stp[1]] = 1
                        self.p[plc] = [0,0]
                        self.board[plc[0],plc[1]] = 0
                    else:
                        p_stp = self.p[plc]
                        self.board[segment[0]+p_stp[0],segment[1]+p_stp[1]] = 2
                        self.board[plc[0],plc[1]] = 1
            self.Snake[index] = [segment[0]+p_stp[0],segment[1]+p_stp[1]]
        change = 0
        if grow_snake:
            self.Snake.append(prev_tail)
            change = 1
        self.num_steps += 1

        if (self.num_steps > (self.food_appeared +1)*self.rate and len(self.foods) < self.max_food):
            x_pos = np.random.randint(0,self.block_size)
            y_pos = np.random.randint(0,self.block_size)
            while self.board[x_pos,y_pos] != 0:
                x_pos = np.random.randint(0,self.block_size)
                y_pos = np.random.randint(0,self.block_size)
            self.board[x_pos,y_pos] = 4
            self.food_appeared += 1

        if not game_on:
            change = -5
        return change,(not game_on)

    def get_board(self):
        return self.board

    def get_total_performance(self):
        return len(self.Snake)
