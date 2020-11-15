import pygame as game
from rimworldenv import SimpleRimWorldEnv


def actionAttack(targetId: int, pos: tuple):
    return targetId, (False, (0, 0)), 0, (True, pos)


def main():
    screen = game.display.set_mode((640, 480))

    env = SimpleRimWorldEnv(1, 1, 10, 10, screen)

    obs = env.reset()

    for i in range(2000):
        obs, rewards, done, info = env.step(actionAttack(0, (9, 9)))
        env.render()

    env.stop()


if __name__ == '__main__':
    main()