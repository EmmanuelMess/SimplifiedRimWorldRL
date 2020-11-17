import numpy as np
import pygame as game
import gym
from gym import spaces


class SimpleRimWorldEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    MAX_ACTORS = 10
    MAX_ENEMIES = 10

    numberOfActors: int
    numberOfEnemies: int

    metadata = {'render.modes': ['human']}
    episodeNumber = 0

    def __init__(self, sizeX: int, sizeY: int, screen: game.Surface = None):
        super(SimpleRimWorldEnv, self).__init__()

        self.sizeX = sizeX
        self.sizeY = sizeY
        self.screen = screen

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.MAX_ACTORS),  #actor select
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

        if actorIndex >= len(self.actors):
            reward -= 0.0000002
        else:
            if isMove:
                if moveTo not in self.actors and moveTo not in self.blocks and moveTo not in self.enemies:
                    self.actors.remove(self.actors[actorIndex])
                    self.actors.append(moveTo)
                else:
                    reward -= 0.0000002

            if isAttack:
                if attackAt in self.enemies:
                    reward += 1.0
                    self.enemies.remove(attackAt)
                else:
                    reward -= 0.0000002

        for enemy in self.enemies:
            if len(self.actors) > 0:
                targetIndex = np.random.randint(0, len(self.actors))
                if np.random.uniform(0, 10) <= 6:
                    self.actors.remove(self.actors[targetIndex])
                    reward -= 1.0

        done = len(self.actors) == 0 or len(self.enemies) == 0
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
        self.episodeNumber += 1

        if self.episodeNumber < 100:
            self.numberOfActors = 1
            self.numberOfEnemies = 1
        elif self.episodeNumber < 200:
            self.numberOfActors = 1
            self.numberOfEnemies = 3
        else:
            self.numberOfActors = 2
            self.numberOfEnemies = 6

        self.actors = [(int(self.sizeX/2), int(self.sizeY/2))] #array of positions
        self.blocks = [] #array of positions
        self.moving = {0: False} #dictionary of index of actor to False or position
        self.enemies = [] #array of positions

        possibleEnemies = [(x, 0) for x in range(self.sizeX)] \
                            + [(x, self.sizeY - 1) for x in range(self.sizeX)] \
                            + [(0, y) for y in range(1, self.sizeY-1)] \
                            + [(self.sizeY-1, y) for y in range(1, self.sizeY-1)]

        np.random.shuffle(possibleEnemies) # FIXME use seed

        self.enemies = possibleEnemies[:self.numberOfEnemies]

        if self.numberOfActors > 1:
            self.actors.append((int(self.sizeX / 2) + 1, int(self.sizeY / 2)))

        return self._getAll()

    def seed(self, seed=None):
        self.seed = seed

    def render(self, mode='human', close=False):
        if self.screen is None:
            return

        for event in game.event.get():
            if event.type == game.QUIT:
                self.screen = None
                game.display.quit()
                return

        self.screen.fill((0, 0, 0))
        game.display.flip()

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
