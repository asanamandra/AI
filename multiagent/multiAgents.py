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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return float("inf")
        else:
            dist = util.manhattanDistance((currentGameState.getGhostPosition(1)), newPos)
            score = max(dist, 2) + successorGameState.getScore()
            nearestfood = 100
            for foodpos in ( newFood.asList()):
                distfood = util.manhattanDistance(foodpos, newPos)
                if (distfood < nearestfood):
                    nearestfood = distfood
            if (currentGameState.getNumFood() > successorGameState.getNumFood()):
                score += 100
            else:
                score -= 2
            if action == Directions.STOP:
                score -= 2
            score -= 2 * nearestfood
            if successorGameState.getPacmanPosition() in (currentGameState.getCapsules()):
                score += 100
            return score
        # return successorGameState.getScore()

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

        move = self.maxAgent(gameState,0)
        return move


    def minAgent(self,state,depth,agent):
        if state.isWin() or state.isLose():
            return state.getScore()
        else:
            next = agent + 1
            if agent == state.getNumAgents()-1:
                next = 0
            minScore =  float("inf")
            for move in state.getLegalActions(agent):
                if next != 0:
                    score = self.minAgent(state.generateSuccessor(agent,move),depth,next)
                else:
                    if depth != self.depth -1:
                        score = self.maxAgent(state.generateSuccessor(agent,move),depth+1)
                    else:
                        score = self.evaluationFunction(state.generateSuccessor(agent,move))
                minScore = min(score,minScore)
            return minScore
    def maxAgent(self,state,depth):
        if state.isLose() or state.isWin():
            return state.getScore()
        else:
            maxScore = float("-inf")
            stopped = Directions.STOP
            for move in state.getLegalActions(0):
                score = self.minAgent(state.generateSuccessor(0,move),depth,1)
                if score > maxScore:
                    maxScore = score
                    stopped = move
            if depth != 0:
                return maxScore
            else:
                return stopped
        # util.raiseNotDefined()
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        move = self.maxAgent(gameState, 0, float("-inf"), float("inf"))
        return move

    def minAgent(self, state, depth, agent, alpha, beta):
        if state.isWin() or state.isLose():
            return state.getScore()
        else:
            next = agent + 1
            if agent == state.getNumAgents() - 1:
                next = 0
            minScore = float('inf')
            for move in state.getLegalActions(agent):
                if next != 0:
                    score = self.minAgent(state.generateSuccessor(agent, move), depth, next, alpha, beta)
                else:
                    if depth != self.depth - 1:
                        score = self.maxAgent(state.generateSuccessor(agent, move), depth + 1, alpha, beta)
                    else:
                        score = self.evaluationFunction(state.generateSuccessor(agent, move))
                minScore = min(score, minScore)
                beta = min(beta, minScore)
                if alpha > minScore:
                    return minScore
            return minScore

    def maxAgent(self, state, depth, alpha, beta):
        if state.isLose() or state.isWin():
            return state.getScore()
        else:
            maxScore = float("-inf")
            stopped = Directions.STOP
            for move in state.getLegalActions(0):
                score = self.minAgent(state.generateSuccessor(0, move), depth, 1, alpha, beta)
                if score > maxScore:
                    stopped = move
                    maxScore = score
                alpha = max(alpha, maxScore)
                if beta < maxScore:
                    return maxScore
            if depth != 0:
                return maxScore
            else:
                return stopped
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectedValue(gameState, agentIndex, depth):
            if depth == 0:
                return self.evaluationFunction(gameState)
            elif gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            else:
                value = 0
                for action in gameState.getLegalActions(agentIndex):
                    if agentIndex != gameState.getNumAgents() - 1:
                        value += expectedValue((gameState.generateSuccessor(agentIndex, action)), agentIndex + 1, depth)
                    else:
                        value += maxvalue((gameState.generateSuccessor(agentIndex, action)), depth - 1)
                return value / len(gameState.getLegalActions(agentIndex))
        def maxvalue(gameState, depth):
            if depth == 0:
                return self.evaluationFunction(gameState)
            elif gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            score = float("-inf")
            for action in  gameState.getLegalActions(0):
                score = max(score, expectedValue((gameState.generateSuccessor(0, action)), 1, depth))
            return score

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        bestaction = Directions.STOP
        score = float("-inf")
        for action in gameState.getLegalActions(0):
            prevscore = score
            score = max(score, expectedValue((gameState.generateSuccessor(0, action)), 1, self.depth))
            if score > prevscore:
                bestaction = action
        return bestaction
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    score = scoreEvaluationFunction(currentGameState)
    food = float("inf")
    for pos in currentGameState.getFood().asList():
        if (util.manhattanDistance(pos,currentGameState.getPacmanPosition()))<food:
            food = util.manhattanDistance(pos,currentGameState.getPacmanPosition())
    numghosts = currentGameState.getNumAgents()-1
    i=1
    disttoghost=float("inf")
    while i <= numghosts:
        disttoghost = min(disttoghost, util.manhattanDistance(currentGameState.getPacmanPosition(),currentGameState.getGhostPosition(i)))
        i+=1
    score += max(disttoghost,3.1)*-8
    score -= food*0.2
    score -= 0.1*len(currentGameState.getFood().asList())
    score -=0.2*len(currentGameState.getCapsules())
    return score
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
