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


from telnetlib import theNULL
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        """Makes sure the action doesn't cause pacman to die, then tries to minimize distance to 
        food and reduce the number of food pellets on the board."""
        
        newGhostLocs = successorGameState.getGhostPositions()

        ghostDistances = []
        for pos in newGhostLocs:
          dist = (abs(newPos[0] - pos[0]) + abs(newPos[1] - pos[1]))
          ghostDistances.append(dist)

        needToMove = False
        curPos = currentGameState.getPacmanPosition()
        if newPos == curPos:
          needToMove = True

        goingToDie = False
        for dist in ghostDistances:
          if dist == 0:
            goingToDie = True

        newFoodLocs = []
        for i in range(newFood.width):
          for j in range(newFood.height):
            if newFood[i][j] == True:
              newFoodLocs.append((i,j))

        if len(newFoodLocs) == 0:
          return 0
        
        totalFoodDistance = 0
        nextToFood = False
        
        newFoodCount = successorGameState.getNumFood()

        for pos in newFoodLocs:
          dist = (abs(newPos[0] - pos[0]) + abs(newPos[1] - pos[1]))
          totalFoodDistance += dist
        avgFoodDist = totalFoodDistance / len(newFoodLocs)

        if goingToDie or needToMove:
          return -1000
        else:
          return 1/avgFoodDist - newFoodCount

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
        """
        "*** YOUR CODE HERE ***"

        def minimaxHelper(state, depth, index):
          # sets values for next recursive call
          nextDepth = depth
          nextIndex = index
          
          legalMoves = state.getLegalActions(index)
          
          # finds terminal states and returns their score
          if state.isWin() or state.isLose() or depth == self.depth or len(legalMoves) == 0:
            return (self.evaluationFunction(state), None)
          
          # if this is the last agent, prepares to recurse over the next depth level, starting back with pacman
          elif index == state.getNumAgents() - 1:
            nextIndex = 0
            nextDepth += 1
          
          # prepares to recurse over the next agent
          else:
            nextIndex += 1

          # finds the move with the maximum score
          if index == 0:
            max = -float("inf")
            bestAction = None
            for i in range(len(legalMoves)):
              tmp = minimaxHelper(state.generateSuccessor(index, legalMoves[i]), nextDepth, nextIndex)[0]
              if tmp > max:
                max = tmp
                bestAction = legalMoves[i]
            return (max, bestAction)

          # finds the move with the minimum score
          else:
            min = float("inf")
            bestAction = None
            for i in range(len(legalMoves)):
              tmp = minimaxHelper(state.generateSuccessor(index, legalMoves[i]), nextDepth, nextIndex)[0]
              if tmp < min:
                min = tmp
                bestAction = legalMoves[i]
            return (min, bestAction)

        return minimaxHelper(gameState, 0, 0)[1]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaHelper(state, depth, index, alpha, beta):
          # same as minimax
          nextDepth = depth
          nextIndex = index
          legalMoves = state.getLegalActions(index)
          if state.isWin() or state.isLose() or depth == self.depth or len(legalMoves) == 0:
            return (self.evaluationFunction(state), None)
          elif index == state.getNumAgents() - 1:
            nextIndex = 0
            nextDepth += 1
          else:
            nextIndex += 1

          # finds the max, accounting for alpha and beta
          if index == 0:
            v = -float("inf")
            bestAction = None
            for i in range(len(legalMoves)):
              tmp = alphaBetaHelper(state.generateSuccessor(index, legalMoves[i]), nextDepth, nextIndex, alpha, beta)[0]
              if tmp > v:
                v = tmp
                bestAction = legalMoves[i]
              if v > beta:
                return (v, bestAction)
              alpha = max(alpha, v)
            return (v, bestAction)

          # finds the min, accounting for alpha and beta
          else:
            v = float("inf")
            bestAction = None
            for i in range(len(legalMoves)):
              tmp = alphaBetaHelper(state.generateSuccessor(index, legalMoves[i]), nextDepth, nextIndex, alpha, beta)[0]
              if tmp < v:
                v = tmp
                bestAction = legalMoves[i]
              if v < alpha:
                return (v, bestAction)
              beta = min(beta, v)
            return (v, bestAction)

        return alphaBetaHelper(gameState, 0, 0, -float("inf"), float("inf"))[1]

        util.raiseNotDefined()

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

        def expectimaxHelper(state, depth, index):
          # same as minimax
          nextDepth = depth
          nextIndex = index
          legalMoves = state.getLegalActions(index)
          if state.isWin() or state.isLose() or depth == self.depth or len(legalMoves) == 0:
            return (self.evaluationFunction(state), None)
          elif index == state.getNumAgents() - 1:
            nextIndex = 0
            nextDepth += 1
          else:
            nextIndex += 1

          # max is the same as minimax
          if index == 0:
            max = -float("inf")
            bestAction = None
            for i in range(len(legalMoves)):
              tmp = expectimaxHelper(state.generateSuccessor(index, legalMoves[i]), nextDepth, nextIndex)[0]
              if tmp > max:
                max = tmp
                bestAction = legalMoves[i]
            return (max, bestAction)

          # picks a random state for the ghosts
          else:
            v = 0
            for i in range(len(legalMoves)):
              p = float(1/len(legalMoves))
              v += p * expectimaxHelper(state.generateSuccessor(index, legalMoves[i]), nextDepth, nextIndex)[0]
            chosenAction = random.choice(legalMoves)
            return (v, chosenAction)


        return expectimaxHelper(gameState, 0, 0)[1]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: I started by making sure that pacman never let himself die if there were any other options.
      Then I prioritized getting closer to food pellets and actively trying to lower the number of pellets.
      Then I noticed that pacman wasn't trying to stay far away from the ghost, so I minimized the average distance
      to the ghosts. Then I added the score to the total so pacman would eat scared ghosts. He never seemed to want to
      eat the last pellet, so I tried putting in a way to force him to end the game if possible, but he still rarely ends
      the game without being pushed by the ghost.
    """
    "*** YOUR CODE HERE ***"
    # This was supposed to make pacman always take the last one but he still doesn't like to.
    if currentGameState.isWin():
      return 1000000

    #getting info from the state
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodCount = currentGameState.getNumFood()
    newGhostLocs = currentGameState.getGhostPositions()

    #keeps track of the distance to each ghost
    ghostDistances = []
    for pos in newGhostLocs:
      dist = (abs(newPos[0] - pos[0]) + abs(newPos[1] - pos[1]))
      ghostDistances.append(dist)
    
    #checks if pacman is going to die. Looking back I probably could've just used state.isLose()
    goingToDie = False
    for dist in ghostDistances:
      if dist == 0:
        goingToDie = True
    
    # finds average distance to a ghost
    totalGhostDistance = sum(ghostDistances)
    avgGhostDistance = totalGhostDistance/len(ghostDistances)

    # keeps track of the locations of all the pellets
    newFoodLocs = []
    for i in range(newFood.width):
      for j in range(newFood.height):
        if newFood[i][j] == True:
          newFoodLocs.append((i,j))
    
    # finds the average distance to a food pellet
    totalFoodDistance = 0
    avgFoodDist = 0
    for pos in newFoodLocs:
      dist = (abs(newPos[0] - pos[0]) + abs(newPos[1] - pos[1]))
      totalFoodDistance += dist
    if len(newFoodLocs) != 0:
      avgFoodDist = totalFoodDistance / len(newFoodLocs)

    # prioritizes not dying
    if not goingToDie:
      return 1/(avgFoodDist) - 2*newFoodCount - 0.5*avgGhostDistance + 1000*currentGameState.getScore()
    else:
      return -1000

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

