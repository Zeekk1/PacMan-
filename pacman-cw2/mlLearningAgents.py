
#importing libraries
from __future__ import absolute_import
from __future__ import print_function

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

import random
from collections import defaultdict

class GameStateFeatures:
    """
    this class facilitates the Q-learning process by getting useful features from the game state
    and returning a hash function to store the state-action pair in the QValue dictionary.

    Attributes:
        loc (tuple): The current location of Pac-Man.
        ghost_loc (tuple): The current location of all ghosts.
        food_map (tuple): The current spread and location of food.
        legalActions (list): The legal actions currently available to Pac-Man (except STOP).

    Args:
        state (GameState): The current game state

    """
    def __init__(self, state: GameState):
    #initializing the state features
        self.loc = state.getPacmanPosition() #pacman's location
        self.ghost_loc = tuple(state.getGhostPositions()) #ghosts' location
        self.food_map = tuple(map(tuple, state.getFood())) #food location
        self.legalActions = state.getLegalPacmanActions()
        
        if Directions.STOP in self.legalActions:
            self.legalActions.remove(Directions.STOP)

    def __hash__(self):
    # using hash function to store the state-action pair in QValue dictionary
        return hash((self.loc, self.ghost_loc, self.food_map))

    def __eq__(self, other):
    # using equality funtion to compare the state-action pair in QValue dictionary
        return self.loc == other.loc and self.ghost_loc == other.ghost_loc \
               and self.food_map == other.food_map

class QLearnAgent(Agent):

    """
    Creating a Q-learning agent for Pac-Man.

    Parameters:
        alpha (float): Rate of learning
        epsilon (float): Rate of exploration
        gamma (float): Factor of discount
        maxAttempts (int): Max tries for each action in a given state.
        numTraining (int): Number of training episodes.

    Attributes:
        QValue (dict): Stores Q-values for each state and action pair.
        lastAction (str): Last action taken by Pac-Man.
        lastState (GameStateFeatures): Last state of game that was encountered.
        visitedTimes (dict): Count of action visits in a given state.
        epsisodesSoFar: the number of games we have played
        maxActionValue: the maximum value of action in a given state
    """

    def __init__(self, alpha: float = 0.2, epsilon: float = 0.01, gamma: float = 0.9,
                 maxAttempts: int = 50, numTraining: int = 10):
    #Initializing the QLearnAgent with given parameters
        
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        
        self.episodesSoFar = 0 
        self.QValue = defaultdict(float) 
        self.lastAction = "East" # Default action to take (East or West)
        self.lastState = None #last state initialized to None
        self.visitedTimes = defaultdict(int) 
        self.maxActionValue = defaultdict(int)

    """---Accessor functions for the number of games played to control learning---"""
    
    def incrementEpisodesSoFar(self):
        # increment the number of episodes played by 1
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        # get the number of episodes played so far
        return self.episodesSoFar

    def getNumTraining(self):
        #get the number of training episodes
        return self.numTraining

    def setEpsilon(self, value: float):
        # Accessor functions for parameters
        self.epsilon = value

    def getAlpha(self) -> float:
        #get alpha value
        return self.alpha

    def setAlpha(self, value: float):
        #set alpha value
        self.alpha = value

    def getGamma(self) -> float:
        #get gamma value
        return self.gamma

    def getMaxAttempts(self) -> int:
        #get maximum attempts
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        """
        Input:
            startState: game state at the start of the game
            endState: resulting game state after the game

        Outputs:
            float: reward comouted based on state transition
        """

        # The reward is the difference in game scores between the endState and startState
        reward = endState.getScore() - startState.getScore()

        # Store visited states in a set to keep track of visited states and penalize oscillation
        endStateFeatures = GameStateFeatures(endState)
        if not hasattr(GameStateFeatures, "visitedStates"):
            GameStateFeatures.visitedStates = set()

        if (endStateFeatures.loc, endStateFeatures.food_map) in GameStateFeatures.visitedStates:
            reward -= 3.0  #penalty for oscillation

        GameStateFeatures.visitedStates.add((endStateFeatures.loc, endStateFeatures.food_map))

        return reward

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Input:
            state: The current state of game state features
            action: The action being evaluated (direction)

        Outputs:
            float: The Q-value connected to the state-action pair
        """

        return self.QValue[(state, action)]

    def getBestAction(self, state: GameStateFeatures):
        """
        Find actions with the most value from legal actions
        if there are multiple best actions with the same value, randomly choose one

        Input:
            state: The current state of game state features

        Outputs:
            bestAction: The best action based on Q-values.
        """

        bestValue = max([self.getQValue(state, a) for a in state.legalActions])
        bestActions = [a for a in state.legalActions if self.getQValue(state, a) == bestValue]
        return random.choice(bestActions)  # Break ties randomly for better exploration


    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Input:
            state: The current state of game state features

        Outputs:
            float: The maximum Q-value (estimated) possible from the state.
        """
        
        if not state.legalActions:
        # If there are no legal actions for Pac-Man in a terminal state, return 0.0
            maxQValue = 0.0
        else:
        # Otherwise, return the maximum Q-value from the legal actions
            maxQValue = max([self.getQValue(state, a) for a in state.legalActions])
        
        return maxQValue


    def learn(self, state: GameStateFeatures, action: Directions,
              reward: float, nextState: GameStateFeatures):
        """
        Input:
            state: The initial state of game state features
            action: The action being taken (direction)
            reward: The received reward (float)
            nextState: The next state of game state features

        Outputs:
            None
        """
        alpha = max(0.1, self.alpha * 0.99)  # Gradually lower rate of learning
        self.alpha = alpha

        QValue_last = self.getQValue(state, action)
        maxQValue_current = self.maxQValue(nextState)

        new_QValue = QValue_last + alpha * (reward + self.getGamma() * maxQValue_current - QValue_last)
        self.QValue[(state, action)] = new_QValue


    def updateCount(self, state: GameStateFeatures, action: Directions):
        """
        Update the visitation count for the state-action pair

        Input:
            state: The current state of game state features.
            action: The action taken in the given game state (Direction).

        Outputs:
            None
        """

        self.visitedTimes[(state, action)] += 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """
        Getting count of the action taken in the given state

        Input:
            state: The current state of game state features.
            action: The action taken in the given game state (Direction).

        Outputs:
            int: The number of times the action has been taken in the given state.
        """
        return self.visitedTimes[(state, action)]


    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Encourage exploration of lesser-visited actions

        Input:
            utility: Value of expected utility in a given state and action (float).
            counts: Counts of the action taken in the given state (int).

        Outputs:
            float: Value of exploration, encouraging exploration of lesser-visited actions.
        """
        
        return utility + 1.0 if counts < self.getMaxAttempts() else utility


    def getAction(self, state: GameState) -> Directions:
        """
        Input:
            state: The current state of game state features.

        Outputs:
            Directions: the chosen action (direction) to take in the given state.
        """

        # If lastState is None, assign the current state to lastState
        if self.lastState is None: self.lastState = state

        # Get the last state and current state features
        lastStateFeatures = GameStateFeatures(self.lastState)
        currentStateFeatures = GameStateFeatures(state)

        # Compute the reward for the last state-action pair
        reward = self.computeReward(self.lastState, state)

        # Update the Q-value for the last state-action pair
        self.learn(lastStateFeatures, self.lastAction, reward, currentStateFeatures)

        if util.flipCoin(self.epsilon):
        # execute the exploration function to encourage exploration of lesser-visited actions
            action = random.choice(currentStateFeatures.legalActions)
        else:
        # else, choose the best action based on Q-values
            action = self.getBestAction(currentStateFeatures)

        # store the last state and action
        self.lastState = state
        self.lastAction = action

        return action

    def final(self, state: GameState):
        """
        Input:
            state: The final state of game state features.

        Outputs:
            None
        """
        reward = self.computeReward(self.lastState, state)

        # Update the Q-value for the last state-action pair
        self.learn(GameStateFeatures(self.lastState), self.lastAction, reward,
                   GameStateFeatures(state))
        
        # Print the game number when the game ends
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Increment the number of games played
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
