import numpy as np
import pygame as game
import gym
from gym import spaces


class SimpleRimWorldEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, numberOfActors: int, numberOfEnemies: int, sizeX: int, sizeY: int, screen: game.Surface = None):
        super(SimpleRimWorldEnv, self).__init__()

        self.numberOfActors = numberOfActors
        self.numberOfEnemies = numberOfEnemies
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.screen = screen

        self.action_space = spaces.Tuple((
            spaces.Discrete(numberOfActors),  #actor select
            spaces.Tuple((spaces.Discrete(2), spaces.Tuple((spaces.Discrete(sizeX), spaces.Discrete(sizeY))))),  #move
            spaces.Discrete(4),  #place block at
            spaces.Tuple((spaces.Discrete(2), spaces.Tuple((spaces.Discrete(sizeX), spaces.Discrete(sizeY))))),#attack
        ))
        self.observation_space = spaces.Box(0, 3, [sizeX, sizeY])#each position can be empty, a box, or an actor

    def step(self, action):
        reward = 0

        #FIXME make moving continuous
        actorIndex = action[0]
        isMove = action[1][0]
        moveTo = action[1][1]
        placeBlockAt = action[2]
        isAttack = action[3][0]
        attackAt = action[3][1]

        if isAttack:
            if attackAt in self.enemies:
                reward += 1.0
                self.enemies.remove(attackAt)

        done = len(self.enemies) == 0
        obs = self._getAll()

        return obs, reward, done, {}

    def _getAll(self):
        return np.asarray([[self._getElemForPos((i, j)) for j in range(self.sizeY)] for i in range(self.sizeX)]).ravel()

    def _getElemForPos(self, pos: tuple):
        if pos in self.actors:
            return 1
        if pos in self.blocks:
            return 2
        if pos in self.enemies:
            return 3
        return 0

    def reset(self):
        self.actors = [(0, 0)] #array of positions
        self.blocks = [] #array of positions
        self.moving = {0: False} #dictionary of index of actor to False or position
        self.enemies = [] #array of positions
        for i in range(min(self.numberOfEnemies, 9)):
            self.enemies.append((9, i))
        return self._getAll()

    def render(self, mode='human', close=False):
        if self.screen is None:
            return

        for event in game.event.get():
            if event.type in (game.QUIT, game.KEYDOWN):
                self.screen = None
                game.display.quit()
                return

        SQUARE_SIZE = min(self.screen.get_rect().width/self.sizeX, self.screen.get_rect().height/self.sizeY)
        WHITE = game.Color(255, 255, 255)
        RED = game.Color(255, 0, 0)

        for actor in self.actors:
            pos = (actor[0]*SQUARE_SIZE + SQUARE_SIZE/2, actor[1]*SQUARE_SIZE + SQUARE_SIZE/2)
            game.draw.circle(self.screen, WHITE, pos, SQUARE_SIZE/2 - 5)

        for box in self.blocks:
            rectangle = (box[0] * SQUARE_SIZE, box[1] * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            game.draw.rect(self.screen, WHITE, rectangle)

        for enemy in self.enemies:
            pos = (enemy[0]*SQUARE_SIZE + SQUARE_SIZE/2, enemy[1]*SQUARE_SIZE + SQUARE_SIZE/2)
            game.draw.circle(self.screen, RED, pos, SQUARE_SIZE/2 - 5)

        game.display.update()

    def stop(self):
        self.screen = None
        game.display.quit()
