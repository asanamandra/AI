# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        state = self.mdp.getStates()[2]
        for i in range(self.iterations):
          temp = self.values.copy()
          for state in self.mdp.getStates():
            finalValue = None
            for action in self.mdp.getPossibleActions(state):
              if finalValue == None or finalValue < self.computeQValueFromValues(state,action):
                finalValue = self.computeQValueFromValues(state,action)
            if finalValue == None:
              finalValue = 0
            else:
                temp[state] = finalValue
          self.values = temp
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state,action):
          value += ((self.discount * self.values[nextState]) + self.mdp.getReward(state, action, nextState)) * probability
        return value
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if len(self.mdp.getPossibleActions(state)) == 0:
          return None
        value = None
        for action in self.mdp.getPossibleActions(state):
          if value == None or self.computeQValueFromValues(state, action) > value:
            value = self.computeQValueFromValues(state, action)
            result = action
        return result
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        values = util.Counter()
        for i in range(self.iterations):
            values = self.values.copy()
            possibleVals = []
            if self.mdp.isTerminal(self.mdp.getStates()[i%len(self.mdp.getStates())]):
                self.values[self.mdp.getStates()[i%len(self.mdp.getStates())]] = 0
            else:
                for action in self.mdp.getPossibleActions(self.mdp.getStates()[i%len(self.mdp.getStates())]):
                    tempValue = 0
                    for t in self.mdp.getTransitionStatesAndProbs(self.mdp.getStates()[i%len(self.mdp.getStates())], action):
                        tempValue += t[1]*(self.mdp.getReward(self.mdp.getStates()[i%len(self.mdp.getStates())], action, t[0]) + self.discount * values[t[0]])
                    possibleVals.append(tempValue)
                self.values[self.mdp.getStates()[i%len(self.mdp.getStates())]] = max(possibleVals)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def theVals(self, state):
        vals = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            vals[action] = self.computeQValueFromValues(state, action)
        return vals

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        prev = dict()
        for state in self.mdp.getStates():
            prev[state]=set()
        for state in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(state):
                for nextState, p in self.mdp.getTransitionStatesAndProbs(state, a):
                    if p>0:
                        prev[nextState].add(state)
        theQ = util.PriorityQueue()
        for state in self.mdp.getStates():
            if len(self.theVals(state)) > 0:
                maxQValue = self.theVals(state)[self.theVals(state).argMax()]
                theQ.push(state, -(abs(self.values[state] - maxQValue)))
        for i in range(self.iterations):
            if theQ.isEmpty():
                return;
            state = theQ.pop()
            maxQValue = self.theVals(state)[self.theVals(state).argMax()]
            self.values[state] = maxQValue
            for p in prev[state]:

                maxQValue = self.theVals(p)[self.theVals(p).argMax()]

                if (abs(self.values[p] - maxQValue)) > self.theta:
                    theQ.update(p, -(abs(self.values[p] - maxQValue)))
