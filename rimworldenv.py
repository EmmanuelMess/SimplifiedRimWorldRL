import numpy as np
import pygame as game
import gym
from gym import spaces

import intersection


class SimpleRimWorldEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    SQUARE_SIZE: float
    WHITE = game.Color(255, 255, 255)
    RED = game.Color(255, 0, 0)

    MAX_ACTORS = 10
    MAX_ENEMIES = 10

    numberOfActors: int
    numberOfEnemies: int

    metadata = {'render.modes': ['human']}
    episodeNumber = 0

    def __init__(self, sizeX: int, sizeY: int, screen: game.Surface = None):
        super(SimpleRimWorldEnv, self).__init__()

        self.SQUARE_SIZE = 0 if (screen is None) else min(screen.get_rect().width / sizeX, screen.get_rect().height / sizeY)

        self.sizeX = sizeX
        self.sizeY = sizeY
        self.screen = screen
        self.shots = []

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

        doneSomething = False

        if actorIndex >= len(self.actors):
            reward -= 0.0000002
        else:
            if isMove:
                if moveTo not in self.actors and moveTo not in self.blocks and moveTo not in self.enemies:
                    self.actors.remove(self.actors[actorIndex])
                    self.actors.append(moveTo)
                    doneSomething = True
                else:
                    reward -= 0.0000002

            if isAttack:
                collided = self._checkCollision(self.actors[actorIndex], attackAt)
                if attackAt in self.enemies and not collided:
                    reward += 1.0
                    self.enemies.remove(attackAt)
                    doneSomething = True
                else:
                    reward -= 0.0000002

        if not doneSomething:
            self.episodesNotDoingAnything += 1
        else:
            self.episodesNotDoingAnything = 0

        for enemy in self.enemies:
            if len(self.actors) > 0:
                targetIndex = np.random.randint(0, len(self.actors))
                target = self.actors[targetIndex]
                if np.random.uniform(0, 10) < 6:
                    collides = self._checkCollision(enemy, target)
                    if not collides:
                        self.shots = [(enemy, target)]
                        self.actors.remove(target)
                        reward -= 1.0

        if len(self.enemies) == 0:
            reward += 0.05
            self._addEnemies()

        if self.episodesNotDoingAnything >= 10:
            reward -= 0.05

        done = len(self.actors) == 0 or self.episodesNotDoingAnything >= 10
        obs = self._getAll()

        return obs, reward, done, {}

    def _checkCollision(self, shooter, target) -> bool:
        shooter = (shooter[0]*self.SQUARE_SIZE + self.SQUARE_SIZE/2, shooter[1]*self.SQUARE_SIZE + self.SQUARE_SIZE/2)
        target = (target[0]*self.SQUARE_SIZE + self.SQUARE_SIZE/2, target[1]*self.SQUARE_SIZE + self.SQUARE_SIZE/2)

        for box in self.blocks:
            box = (box[0]*self.SQUARE_SIZE, box[1]*self.SQUARE_SIZE)
            if intersection.doesIntersect(shooter, target, box, 1*self.SQUARE_SIZE, 1*self.SQUARE_SIZE):
                return True

        return False

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

    def _addEnemies(self):
        possibleEnemies = [(x, 0) for x in range(self.sizeX)] \
                            + [(x, self.sizeY - 1) for x in range(self.sizeX)] \
                            + [(0, y) for y in range(1, self.sizeY-1)] \
                            + [(self.sizeY-1, y) for y in range(1, self.sizeY-1)]

        np.random.shuffle(possibleEnemies) # FIXME use seed

        self.enemies = possibleEnemies[:self.numberOfEnemies]

    def reset(self):
        self.episodeNumber += 1

        if self.episodeNumber < 5000:
            self.numberOfActors = 1
            self.numberOfEnemies = 1
        else:
            self.numberOfActors = 2
            self.numberOfEnemies = 3

        self.actors = [(int(self.sizeX/2), int(self.sizeY/2))] #array of positions
        self.blocks = [(1, 1)] #array of positions
        self.moving = {0: False} #dictionary of index of actor to False or position
        self.enemies = [] #array of positions

        self.episodesNotDoingAnything = 0

        self._addEnemies()

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

        for actor in self.actors:
            pos = (actor[0]*self.SQUARE_SIZE + self.SQUARE_SIZE/2, actor[1]*self.SQUARE_SIZE + self.SQUARE_SIZE/2)
            game.draw.circle(self.screen, self.WHITE, pos, self.SQUARE_SIZE/2 - 5)

        for box in self.blocks:
            rectangle = (box[0] * self.SQUARE_SIZE, box[1] * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE)
            game.draw.rect(self.screen, self.WHITE, rectangle)

        for enemy in self.enemies:
            pos = (enemy[0]*self.SQUARE_SIZE + self.SQUARE_SIZE/2, enemy[1]*self.SQUARE_SIZE + self.SQUARE_SIZE/2)
            game.draw.circle(self.screen, self.RED, pos, self.SQUARE_SIZE/2 - 5)

        for shot in self.shots:
            game.draw.line(self.screen, self.WHITE,
                           game.Vector2(shot[0]) * self.SQUARE_SIZE + game.Vector2(self.SQUARE_SIZE/2, self.SQUARE_SIZE/2),
                           game.Vector2(shot[1]) * self.SQUARE_SIZE + game.Vector2(self.SQUARE_SIZE/2, self.SQUARE_SIZE/2))

        self.shots = []

        game.display.update()

    def stop(self):
        self.screen = None
        game.display.quit()
