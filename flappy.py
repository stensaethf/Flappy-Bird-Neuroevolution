# flappy.py
# Flappy bird game with evolutionary neural networks.
# Author: sourabhv
# Modified by Frederik Roenn Stensaeth
# 05.23.17

from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *
import neuralNetwork
import numpy as np
from copy import deepcopy

FPS = 30
SCREENWIDTH  = 288.0
SCREENHEIGHT = 512.0
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image and hitmask dicts
IMAGES, HITMASKS = {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

crossover_layer = True
crossover_rate = 0.15
difficulty_game = 2
fitness_boost = 25
generation = 0
mutation_rate = 0.15
num_elites = 2
perfect_game = False
pipe_next_dist = None
pipe_next_height = None
population = []
pop_fitness = []
pop_size = 50
save_gen = 1
strategy = 0
structure = [3, 7, 1]

# Initialize population.
for i in range(pop_size):
    # Create neural net individual.
    individual = neuralNetwork.NeuralNetwork(structure)
    population.append(individual)
    pop_fitness.append(0) # every inidividual starts with a fitness of 0.

##### TODO #####
# 0. tanh
# 1. NN --> create dataset where if we are lower than the hole, flap. If above,
#    not flap. --> can i create this dataset in excel? randomly generate holes
#    and position of bird. assign flap/no flap values to the sets.
##### END #####

def main():
    global SCREEN, FPSCLOCK
    global pop_fitness
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((int(SCREENWIDTH), int(SCREENHEIGHT)))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = {
            'basex': 0,
            'playerIndexGen': cycle([0, 1, 2, 1]),
            'playery': int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2),
        }

        mainGame(movementInfo)
        showGameOverScreen()

def mainGame(movementInfo):
    global pop_fitness
    global pipe_next_height
    global pipe_next_dist
    global FPS
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    # each individual in the population needs a x and y position.
    individuals_x, individuals_y = [], []
    for i in range(pop_size):
        x, y = int(SCREENWIDTH * 0.2), movementInfo['playery']
        individuals_x.append(x)
        individuals_y.append(y)
    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper and lower pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
    ]
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
    ]
    if not perfect_game:
        upperPipes.append({'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']})
        lowerPipes.append({'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']})

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    individuals_vel_y = [-9] * pop_size # player's velocity along Y, default same as playerFlapped
    playerMaxVelY = 10 # max vel along Y, max descend speed
    playerMinVelY = -8 # min vel along Y, max ascend speed
    individuals_acc_y = [1] * pop_size # players downward accleration
    playerFlapAcc = -9 # players speed on flapping
    individuals_flapped = [False] * pop_size # True when player flaps
    individuals_state = [True] * pop_size # whether a bird has died or not.
    num_alive = pop_size # number of individuals alive

    # Find distance and height of first pipe.
    pipe_next_height = (lowerPipes[0]['y'] + (upperPipes[0]['y'] + IMAGES['pipe'][0].get_height())) / 2
    pipe_next_dist = lowerPipes[0]['x']

    while True:
        # if quit or escape was hit, exit the game.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_UP:
                FPS += 10
            elif event.type == KEYDOWN and event.key == K_DOWN:
                FPS -= 10

        # check for individuals that crashed with the ground. set them to dead.
        for i in range(pop_size):
            if individuals_y[i] < 0 and individuals_state[i] == True:
                num_alive -= 1
                individuals_state[i] = False

        # if all individuals are dead, return.
        if num_alive == 0:
            return

        # update fitness for alive individuals and determine whether individuals
        # will flap or not.
        pipe_next_dist += pipeVelX
        for i in range(pop_size):
            if individuals_state[i] == True:
                pop_fitness[i] += 1
                action = getAction(i, pipe_next_dist, individuals_y[i], pipe_next_height)
                if action == 1 and individuals_y[i] > -2 * IMAGES['player'][0].get_height():
                    individuals_vel_y[i] = playerFlapAcc
                    individuals_flapped[i] = True

        # check for crash here
        crashTest = checkCrash(
            {
                'index': playerIndex,
                'x': individuals_x,
                'y': individuals_y,
            },
            upperPipes,
            lowerPipes,
        )

        # check if individuals crashed. if crashed, dead.
        for i in range(pop_size):
            if individuals_state[i] == True and crashTest[i] == True:
                num_alive -= 1
                individuals_state[i] = False

        # again, check if all individuals are dead.
        if num_alive == 0:
            return

        # check for score and boost fitness of individuals that made it passed
        # a set of pipes.
        passed_pipe = False
        for i in range(pop_size):
            if individuals_state[i] == True:
                playerMidPos = individuals_x[i]
                j = -1
                for pipe in upperPipes:
                    j += 1
                    pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width()
                    if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                        # middle of next pipe hole
                        pipe_next_height = (lowerPipes[j + 1]['y'] + (upperPipes[j + 1]['y'] + IMAGES['pipe'][j + 1].get_height())) / 2
                        # distance to next pipe
                        pipe_next_dist = upperPipes[j + 1]['x']
                        pop_fitness[i] += fitness_boost
                        passed_pipe = True
        if passed_pipe:
            score += 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        for i in range(pop_size):
            if individuals_state[i] == True:
                if individuals_vel_y[i] < playerMaxVelY and not individuals_flapped[i]:
                    individuals_vel_y[i] += individuals_acc_y[i]
                if individuals_flapped[i]:
                    individuals_flapped[i] = False
                playerHeight = IMAGES['player'][playerIndex].get_height()
                individuals_y[i] += min(individuals_vel_y[i], BASEY - individuals_y[i] - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        showAlive(num_alive)
        showGen()
        # update visual position of individuals.
        for i in range(pop_size):
            if individuals_state[i] == True:
                SCREEN.blit(IMAGES['player'][playerIndex], (individuals_x[i], individuals_y[i]))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def showGameOverScreen():
    # Instead of showing the game over screen, we will now perform our
    # selection, mutation and crossover.
    global pop_fitness
    global generation
    global population

    # FITNESS PROPORTIONAL SELECTION
    new_gen = []

    # Make elite clones.
    elites = []
    if num_elites <= pop_size:
        for i in range(num_elites):
            # Find max fitness individual and add it to the next generation.
            elite_index = -1
            for j in range(pop_size):
                if j not in elites and (elite_index == -1 or pop_fitness[j] > pop_fitness[elite_index]):
                    elite_index = j

            if elite_index == -1:
                print 'Invalid elite clone.'
                sys.exit()

            elites.append(elite_index)

        for i in elites:
            # new_gen.append(population[i].getWeightsAndBias())
            elite_copy = neuralNetwork.NeuralNetwork(population[i].getStructure(), True)
            w, b = population[i].getWeightsAndBias()
            elite_copy.setWeightsAndBias(w, b)
            new_gen.append(elite_copy)

    total_fitness = float(sum(pop_fitness))
    prop_fitness = []
    for i in range(pop_size):
        prop_fitness.append(pop_fitness[i] / total_fitness)
        if i > 0:
            prop_fitness[i] += prop_fitness[i - 1]

    # Select parents and create offspring by performing crossover and
    # mutation.
    while len(new_gen) < pop_size:
        parent_percent_1 = random.uniform(0, 1)
        parent_percent_2 = random.uniform(0, 1)
        parent1 = None
        parent2 = None
        # Find a pair of parents.
        for i in range(pop_size):
            if parent_percent_1 <= prop_fitness[i] and parent1 == None:
                parent1 = population[i]
            if parent_percent_2 <= prop_fitness[i] and parent2 == None:
                parent2 = population[i]

            if parent1 != None and parent2 != None:
                break

        # Now that we have selected parents, we do crossover and mutation.
        if strategy == 0:
            values1, values2 = crossover(parent1, parent2)
            values1 = mutate(values1)
            values2 = mutate(values2)
        else:
            values1 = parent1.getWeightsAndBias()
            values2 = parent2.getWeightsAndBias()

        # Add new individuals to the next generation.
        parent1_new = neuralNetwork.NeuralNetwork(parent1.getStructure(), True)
        parent2_new = neuralNetwork.NeuralNetwork(parent2.getStructure(), True)
        parent1_new.setWeightsAndBias(values1[0], values1[1])
        parent2_new.setWeightsAndBias(values2[0], values2[1])
        # new_gen.append(values1)
        # new_gen.append(values2)
        new_gen.append(parent1_new)
        new_gen.append(parent2_new)

    # Check if we have too many individuals in the next generation. This can
    # be the case because we added both offspring to the population.
    if len(new_gen) != pop_size:
        new_gen = new_gen[:-1]

    # reset fitness and update the population of the neural networks.
    for i in range(pop_size):
        pop_fitness[i] = 0
        population[i] = new_gen[i]

    generation += 1

    # now that we have updated the fitnesses, we save the population.
    if generation % save_gen == 0:
        savePopulation()

    return

def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    if difficulty_game == 0:
        gapY = 0
    elif difficulty_game == 1:
        gapY = random.randrange(0, int(BASEY * 0.4 - PIPEGAPSIZE))
    else:
        gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def showAlive(alive):
    """
    Displays the number of birds still alive.
    """
    aliveDigits = [int(x) for x in list(str(alive))]

    Xoffset = 10

    for digit in aliveDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.9))
        Xoffset += IMAGES['numbers'][digit].get_width()

def showGen():
    """
    Displays the generation number.
    """
    global generation
    genDigits = [int(x) for x in list(str(generation))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in genDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = SCREENWIDTH - totalWidth - 10
    for digit in genDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.9))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(players, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    individuals_crash = [False] * pop_size

    for i in range(pop_size):
        pi = players['index']
        players['w'] = IMAGES['player'][0].get_width()
        players['h'] = IMAGES['player'][0].get_height()

        # if player crashes into ground
        if players['y'][i] + players['h'] >= BASEY - 1:
            individuals_crash[i] = True

        # XX do i need this else?
        else:
            playerRect = pygame.Rect(players['x'][i], players['y'][i],
                          players['w'], players['h'])
            pipeW = IMAGES['pipe'][0].get_width()
            pipeH = IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = HITMASKS['player'][pi]
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    individuals_crash[i] = True

    return individuals_crash

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def getAction(num, distance, bird_height, hole_height):
    """
    Gets the action from a network by feeding forward the inputs.
    """
    if strategy == 0:
        global population
        distance = distance / 450 - 0.5
        bird_height = min(bird_height, SCREENHEIGHT) / SCREENHEIGHT - 0.5
        hole_height = min(hole_height, SCREENHEIGHT) / SCREENHEIGHT - 0.5
        # bird_height = float(bird_height) / SCREENHEIGHT
        # hole_height = float(hole_height) / SCREENHEIGHT
        layers = population[num].getStructure()
        if layers[0] == 3:
            inputs = np.reshape([distance, bird_height, hole_height], (3, 1))
        elif layers[0] == 2:
            inputs = np.reshape([bird_height, hole_height], (2, 1))
        else:
            print 'Invalid first layer of neural network.'
            sys.exit()
        thresholds, acts = population[num].feedForward(inputs)
        result = acts[-1][0][0]

        if result > 0.5:
            # Flap.
            return 1
        # Do not flap.
        return 0
    elif strategy == 1:
        # Always flap.
        return 1
    elif strategy == 2:
        # Never flap.
        return 0
    elif strategy == 3:
        # Random flap.
        if random.uniform(0, 1) > 0.5:
            return 1
        else:
            return 0
    else:
        print 'Invalid strategy.'
        sys.exit()

def savePopulation():
    """
    Saves each individual in the population by pickling the network.
    """
    global population
    for i in range(pop_size):
        filename = "models/FlappyBirdModel" + str(i) + ".pickle"
        population[i].saveWeightsAndBias(filename)

def mutate(values):
    """
    Mutates the weights and biases of the network according to a given
    rate.
    """
    weights, bias = values
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if random.uniform(0, 1) <= mutation_rate:
                # Mutate bias.
                bias[i][j][0] += random.uniform(-0.5, 0.5)
            for k in range(len(weights[i][j])):
                if random.uniform(0, 1) <= mutation_rate:
                    # Mutate weight.
                    weights[i][j][k] += random.uniform(-0.5, 0.5)

    return [weights, bias]

def crossover(ind1, ind2):
    """
    Crossover the two networks.
    """
    # Store weights/bias temporarily.
    weights1, bias1 = ind1.getWeightsAndBias()
    weights2, bias2 = ind2.getWeightsAndBias()

    if random.uniform(0, 1) <= crossover_rate:
        if crossover_layer:
            # Swap weights/bias.
            weights1[0], weights2[0] = weights2[0], weights1[0]
            bias1[0], bias2[0] = bias2[0], bias1[0]
        else:
            values1, values2 = crossoverSection(ind1, ind2)
            weights1, bias1 = values1
            weights2, bias2 = values2

    return [weights1, bias1], [weights2, bias2]

def crossoverSection(ind1, ind2):
    """
    Crossover a random section of two networks.
    """
    weights1, bias1 = ind1.getWeightsAndBias()
    weights2, bias2 = ind2.getWeightsAndBias()
    structure1 = ind1.getStructure()
    structure2 = ind2.getStructure()

    # Convert neural network to a single array.
    combined1 = []
    for i in range(len(structure1) - 1):
        for j in weights1[i]:
            combined1 = np.concatenate([combined1, j])
        for j in bias1[i]:
            combined1 = np.concatenate([combined1, j])

    # Convert neural network to a single array.
    combined2 = []
    for i in range(len(structure2) - 1):
        for j in weights2[i]:
            combined2 = np.concatenate([combined2, j])
        for j in bias2[i]:
            combined2 = np.concatenate([combined2, j])

    # Find the max crossover length. We crossover at most half the
    # weights/bias.
    length = len(combined1)
    if length > len(combined2):
        length = len(combined2)
    max_crossover = length / 2

    # Find the sections we want to crossover.
    crossover_l = random.randint(0, max_crossover)
    crossover_p1 = random.randint(0, len(combined1) - crossover_l)
    crossover_p2 = random.randint(0, len(combined2) - crossover_l)
    section1 = deepcopy(combined1[crossover_p1:crossover_p1 + crossover_l])
    section2 = deepcopy(combined2[crossover_p2:crossover_p2 + crossover_l])

    # Crossover the sections.
    combined1[crossover_p1:crossover_p1 + crossover_l] = section2
    combined2[crossover_p2:crossover_p2 + crossover_l] = section1

    # Now we convert the arrays back into layers.
    values1_new = convertArrayToLayers(structure1, combined1)
    values2_new = convertArrayToLayers(structure2, combined2)

    return values1_new, values2_new

def convertArrayToLayers(structure, combined):
    """
    Converts an array of weights and biases into separate lists of matrices
    of weights and biases.
    """
    weights_new = []
    bias_new = []
    index = 0
    for i in range(len(structure) - 1):
        index_weight = index + structure[i] * structure[i + 1]
        weights_new.append(combined[index:index_weight].reshape(structure[i + 1], structure[i]))
        index_bias = index_weight + structure[i + 1]
        bias_new.append(combined[index_weight:index_bias].reshape(structure[i + 1], 1))
        index = index_bias

    return [weights_new, bias_new]

def initialize():
    """
    Initializes the population.
    """
    global pop_size, population, pop_fitness, structure
    # Initialize population.
    population = []
    pop_fitness = []
    for i in range(pop_size):
        # Create neural net individual.
        if structure != None:
            individual = neuralNetwork.NeuralNetwork(structure)
        else:
            s = genRandomStructure()
            individual = neuralNetwork.NeuralNetwork(s)
        population.append(individual)
        pop_fitness.append(0) # every inidividual starts with a fitness of 0.

def genRandomStructure():
    """
    Generates a random neural network structure.
    """
    s = []
    # Input layer has either 2 or three nodes.
    if random.uniform(0, 1) > 0.5:
        s.append(3)
    else:
        s.append(2)

    p = 1.0
    while len(s) < 7 and p > 0.5:
        # Add a new layer of random size to the network.
        # Max layer size is 10 nodes.
        layer_size = random.randint(1, 10)
        s.append(layer_size)
        p = random.uniform(0, 1)

    # Output layer has a single node.
    s.append(1)
    return s

if __name__ == '__main__':
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-fps':
            # Frames per second.
            # Default: 60.
            FPS = int(sys.argv[i + 1])
            i += 1
        elif arg == '-g':
            # Save every x generations.
            # Default: 1.
            save_gen = int(sys.argv[i + 1])
            i += 1
        elif arg == '-m':
            # Mutation rate.
            # Default: 0.05.
            mutation_rate = float(sys.argv[i + 1])
            i += 1
        elif arg == '-c':
            # Crossover rate.
            # Default: 0.15.
            crossover_rate = float(sys.argv[i + 1])
            i += 1
        elif arg == '-e':
            # Number of elite clones.
            # Default: 2.
            num_elites = int(sys.argv[i + 1])
            i += 1
        elif arg == '-p':
            # Population size (required: x >= 2).
            # Default: 50.
            pop_size = int(sys.argv[i + 1])
            i += 1
        elif arg == '-s':
            # Strategy:
            #   0 --> neural net.
            #   1 --> always flap.
            #   2 --> never flap.
            #   3 --> random flap.
            # Default: 0.
            strategy = int(sys.argv[i + 1])
            i += 1
        elif arg == '-n':
            # Different neural network structures.
            # Default: [3, 7, 1].
            structure = [int(x) for x in sys.argv[i + 1].split('/')]
            # Input layer needs 2 or 3 nodes and output layer can only have
            # a single node.
            if structure[0] not in [2, 3] or structure[-1] != 1:
                print 'Invalid command line arguments.'
                sys.exit()
            i += 1
        elif arg == '-l':
            # Game difficulty.
            # Default: 2 (hard).
            difficulty_game = int(sys.argv[i + 1])
            if difficulty_game < 0 or difficulty_game > 2:
                print 'Invalid command line arguments.'
                sys.exit()
            i += 1
        elif arg == '-b':
            # Fitness boost for passing through a set of pipes.
            # Default: 25.
            fitness_boost = int(sys.argv[i + 1])
            i += 1
        elif arg == '--random-nn':
            # Neural network of various structures.
            # Crossover layer will not work, so that is turned off by default.
            crossover_layer = False
            structure = None
        elif arg == '--crossover-section':
            # Change the way crossover is done. Normally we simply swap the
            # weights/bias of the first hidden layer, but now we swap a section
            # of all the weights/bias. The networks will have the same
            # structure as before after crossover.
            crossover_layer = False
        elif arg == '--perfect':
            # Increased distance between pipes so that it is possible to
            # guarantee a perfect game (never die).
            perfect_game = True
        else:
            print 'Invalid command line arguments.'
            sys.exit()
        i += 1

    # Population size or structure was changed, so we need to redo
    # the inialization.
    if pop_size != 50 or structure != [3, 7, 1]:
        initialize()
    main()
