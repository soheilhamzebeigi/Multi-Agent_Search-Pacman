# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        New_States = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in New_States]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        New_Foods_States = newFood.asList()
        Nearing_Ghosts = New_States[0].configuration.pos
        Dist_of_foods = [manhattanDistance(newPos, foodPosition) for foodPosition in New_Foods_States]
        Nearest_Ghost = manhattanDistance(newPos, Nearing_Ghosts)
        score = 0
        if len(Dist_of_foods) == 0:
            return 0
        Nearing_foods = min(Dist_of_foods)
        if action == 'Stop':
            score -= 50
        return successorGameState.getScore() + Nearest_Ghost / (Nearing_foods * 10) + score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.constant_depth = int(depth)
        self.level = 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        Game_Res = self.Value_Catching(gameState, 0, 0)
        return Game_Res[1]

    def Find_The_Minimum_Vale(self, gameState, index, depth):
        Less_acting = ""
        minimum = float("inf")
        Expected_moving = gameState.getLegalActions(index)
        for j in Expected_moving:
            successor = gameState.generateSuccessor(index, j)
            Index_of_current_successor = index + 1
            Depth_of_current_successor = depth
            if Index_of_current_successor == gameState.getNumAgents():
                Index_of_current_successor = 0
                Depth_of_current_successor += 1
            current_value = self.Value_Catching(successor, Index_of_current_successor, Depth_of_current_successor)[0]
            if current_value < minimum:
                minimum = current_value
                Less_acting = j
        return minimum, Less_acting


    def Find_The_Maximum_Vale(self, gameState, index, depth):
        Action_in_Maximum_state = ""
        Value_in_Maximum_state = float("-inf")
        moving = gameState.getLegalActions(index)
        for v in moving:
            successor = gameState.generateSuccessor(index, v)
            Index_of_current_successor = index + 1
            Depth_of_current_successor = depth
            if Index_of_current_successor == gameState.getNumAgents():
                Index_of_current_successor = 0
                Depth_of_current_successor += 1
            Value_of_current_successor = self.Value_Catching(successor, Index_of_current_successor, Depth_of_current_successor)[0]
            if Value_of_current_successor > Value_in_Maximum_state:
                Value_in_Maximum_state = Value_of_current_successor
                Action_in_Maximum_state = v
        return Value_in_Maximum_state, Action_in_Maximum_state

    def Value_Catching(self, gameState, index, depth):
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), ""
        if index == 0:
            return self.Find_The_Maximum_Vale(gameState, index, depth)
        else:
            return self.Find_The_Minimum_Vale(gameState, index, depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    def Find_The_Minimum_Vale(self, game_state, index, depth, alpha, beta):
        Lowest_action = ""
        Lowest_value = float("inf")
        Moves = game_state.getLegalActions(index)
        for action in Moves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1
            Act_in_Now, Ratio = self.Best_Action_and_Best_rate(successor, successor_index, successor_depth, alpha, beta)
            if Ratio < Lowest_value:
                Lowest_value = Ratio
                Lowest_action = action
            beta = min(beta, Lowest_value)
            if Lowest_value < alpha:
                return Lowest_action, Lowest_value

        return Lowest_action, Lowest_value

    def Best_Action_and_Best_rate(self, game_state, index, depth, alpha, beta):
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return "", game_state.getScore()
        if index == 0:
            return self.Find_The_Maximum_Vale(game_state, index, depth, alpha, beta)
        else:
            return self.Find_The_Minimum_Vale(game_state, index, depth, alpha, beta)

    def Find_The_Maximum_Vale(self, game_state, index, depth, alpha, beta):
        Highest_Action = ""
        Highest_value = float("-inf")
        Moves = game_state.getLegalActions(index)
        for action in Moves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1
            Act_in_now, Ratio = self.Best_Action_and_Best_rate(successor, successor_index, successor_depth, alpha, beta)
            if Ratio > Highest_value:
                Highest_value = Ratio
                Highest_Action = action
            alpha = max(alpha, Highest_value)
            if Highest_value > beta:
                return Highest_Action, Highest_value
        return Highest_Action, Highest_value

    def getAction(self, game_state):
        result = self.Best_Action_and_Best_rate(game_state, 0, 0, float("-inf"), float("inf"))
        return result[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, game_state):
        """
        Returns the expectimax acts using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        acts, rate = self.Value_Catching(game_state, 0, 0)
        return acts

    def Find_The_Value(self, game_state, index, depth):
        Value_is_Expecting = 0
        Expected_moving = game_state.getLegalActions(index)
        Prob_of_current_successor = 1.0 / len(Expected_moving)
        The_Action_is_Expecting = ""
        for s in Expected_moving:
            successor = game_state.generateSuccessor(index, s)
            Index_of_current_successor = index + 1
            Depth_of_current_successor = depth
            if Index_of_current_successor == game_state.getNumAgents():
                Index_of_current_successor = 0
                Depth_of_current_successor += 1
            Act_in_now, Ratio = self.Value_Catching(successor, Index_of_current_successor, Depth_of_current_successor)
            Value_is_Expecting += Prob_of_current_successor * Ratio

        return The_Action_is_Expecting, Value_is_Expecting

    def Find_The_Maximum_Vale(self, game_state, index, depth):
        Best_of_actions = ""
        max = float("-inf")
        Expected_moving = game_state.getLegalActions(index)
        for s in Expected_moving:
            successor = game_state.generateSuccessor(index, s)
            Index_of_current_successor = index + 1
            Depth_of_current_successor = depth
            if Index_of_current_successor == game_state.getNumAgents():
                Index_of_current_successor = 0
                Depth_of_current_successor += 1
            Act_in_now, Ratio = self.Value_Catching(successor, Index_of_current_successor, Depth_of_current_successor)
            if Ratio > max:
                max = Ratio
                Best_of_actions = s

        return Best_of_actions, max

    def Value_Catching(self, game_state, index, depth):
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return "", self.evaluationFunction(game_state)
        if index == 0:
            return self.Find_The_Maximum_Vale(game_state, index, depth)
        else:
            return self.Find_The_Value(game_state, index, depth)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    Rate_of_Game = currentGameState.getScore()
    number_of_capsule = len(currentGameState.getCapsules())
    List_of_Foods = currentGameState.getFood().asList()
    number_of_Foods = len(List_of_Foods)
    Nearing_Food = 1
    Positions_of_ghosts = currentGameState.getGhostPositions()
    Position = currentGameState.getPacmanPosition()
    Distance_of_Food = [manhattanDistance(Position, food_position) for food_position in List_of_Foods]

    if number_of_Foods > 0:
        Nearing_Food = min(Distance_of_Food)

    for ghost_position in Positions_of_ghosts:
        ghost_distance = manhattanDistance(Position, ghost_position)
        if ghost_distance < 2:
            Nearing_Food = 99999

    Attribute = [1.0 / Nearing_Food, Rate_of_Game, number_of_Foods, number_of_capsule]
    Measure = [10, 200, -100, -10]
    return sum([feature * weight for feature, weight in zip(Attribute, Measure)])

better = betterEvaluationFunction
