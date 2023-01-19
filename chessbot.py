import numpy as np
import gym
import gym_chess
import random
import copy
import matplotlib.pyplot as plt
import time


#state, reward, done, info
class StepData:
  def __init__(self, state, reward, done, info):
    self.state = state
    self.reward = reward
    self.done = done
    self.info = info
#Nodes of tree
class Node:
  def __init__(self, gameInfo, visits, totalScore , parent_node, action):
    self.gameInfo = gameInfo
    self.visits = visits
    self.totalScore = totalScore
    self.next_node = []
    self.parent_node = parent_node
    self.action = action
    
  def set_next_node(self, next_node):
    self.next_node.append(next_node)
    
  def get_next_node(self):
    return self.next_node
    
  def set_parent_node(self, parent_node):
    self.parent_node = parent_node
    
  def get_parent_node(self):
    return self.parent_node

  def set_totalScore(self, totalScore):
     self.totalScore = totalScore

  def get_totalScore(self):
    return self.totalScore

  def set_visits(self, visits):
     self.visits = visits

  def get_visits(self):
    return self.visits
  
  def get_gameInfo(self):
    return self.gameInfo

  def set_action(self,action):
    self.action = action
  
  def get_action(self):
    return self.action

class Tree:
    def __init__(self, startNode):
        self.startNode = startNode

    def findBestAction(self, s1, roundNr):## return index of the best move in the discrete action space
        if s1.get_next_node() == []:
          print("This node has no leafNodes")
          return -1
        totalScoreToBeat = float("-inf")
        ##Gets nnullType error
        for i in s1.get_next_node():
          #print("Action: " + str(i.get_action()) + " Value: " + str(i.get_totalScore()/i.get_visits()) + " Visits: " + str(i.get_visits()))
          if(i.get_totalScore()/i.get_visits()>= totalScoreToBeat):
            totalScoreToBeat  = i.get_totalScore()/i.get_visits()
            bestNode = i
        return bestNode.get_action()

    def findWorstAction(self, s1, roundNr):## return index of the best move in the discrete action space
        if s1.get_next_node() == []:
          print("This node has no leafNodes")
          return -1
        totalScoreToBeat = float("inf")
        ##Gets nnullType error
        for i in s1.get_next_node():
          #print("Action: " + str(i.get_action()) + " Value: " + str(i.get_totalScore()/i.get_visits()) + " Visits: " + str(i.get_visits()))
          if(i.get_totalScore()/i.get_visits()< totalScoreToBeat):
            totalScoreToBeat  = i.get_totalScore()/i.get_visits()
            bestNode = i
        return bestNode.get_action()
    
    def get_start_node(self):
      return self.startNode

    def findNodeBasedOnActions(self, actions):
      itNode = self.get_start_node()
      for a in actions:
        for next in itNode.get_next_node():
          if(next.get_action() == a):
            itNode = next
            break
      return itNode


    ##remember to backpropagate the value we get here. UPDATE THE TREE
    def rollout(self, s1, gameEnv): ## Implement neural network
        simOfGame = copy.deepcopy(gameEnv)
        iterationNode = s1
        while True:
            if iterationNode.get_gameInfo().done:
              return iterationNode.get_gameInfo().reward
            else:
              a = simOfGame.legal_moves[random.randrange(0,len(simOfGame.legal_moves))]
              state, reward, done, info = simOfGame.step(a)
              iterationNode = Node(StepData(state, reward, done, info),0,0,None, a)

    def backProp(self, s1, value, stopNode):
        while True:
            s1.set_totalScore(s1.get_totalScore() + value)
            s1.set_visits(s1.get_visits() + 1)
            if s1 == stopNode:
              break
            s1 = s1.get_parent_node()
            
    def expandingTree(self, s1, iteratorNr, gameEnv, blacksTurn):
        #getting leafNode:
        startOfExplorationNode = s1
        cloneOfGameEnv = copy.deepcopy(gameEnv)
        finishedTheGame =False
         
        while s1.get_next_node() != []:
            if(blacksTurn):
              maxUcb1 = ucb1Black(s1.get_next_node()[0],iteratorNr)
            else:
              maxUcb1 = ucb1White(s1.get_next_node()[0],iteratorNr)
            newS1 = s1.get_next_node()[0]
            for n in s1.get_next_node():
              if blacksTurn:
                newUcb1 = ucb1Black(n,iteratorNr)
              else:
                newUcb1 = ucb1White(n,iteratorNr)
              if newUcb1 >= maxUcb1:
                maxUcb1 = newUcb1
                newS1 = n
            s1 = newS1
            meow, ewom , finishedTheGame, thingy = cloneOfGameEnv.step(newS1.get_action())

        #rollingout if the leafnode has not been visited
        #Might be useless if i run each leafnoede at inits
        if s1.get_visits() == 0:
            self.backProp(s1, self.rollout(s1, cloneOfGameEnv),startOfExplorationNode)
            return

        # if not terminal state, add new nodes to s1, and roolout for the fist action
        if(finishedTheGame):
          return

        for i in range(0, len(cloneOfGameEnv.legal_moves)): #all moves 
          simOfGame = copy.deepcopy(cloneOfGameEnv)
          state, reward, done, info = simOfGame.step(cloneOfGameEnv.legal_moves[i])
          newChild = Node(StepData(state, reward, done, info),0,0,s1,cloneOfGameEnv.legal_moves[i])
          s1.set_next_node(newChild)
          self.backProp(newChild, self.rollout(newChild,cloneOfGameEnv),startOfExplorationNode)


        return  

        
#Calc used for tree tranversion
def ucb1Black(node,iterationNr):
    if node.get_visits() == 0 or iterationNr == 0:
        return float('inf')
    return node.get_totalScore()/node.get_visits() + explorationCoefficent *(np.sqrt(np.log(iterationNr)/node.get_visits()))

def ucb1White(node,iterationNr):
    if node.get_visits() == 0 or iterationNr == 0:
        return float('inf')
    return -node.get_totalScore()/node.get_visits() + explorationCoefficent *(np.sqrt(np.log(iterationNr)/node.get_visits()))



games = []
wins = []
explorationRuns= 5


explorationCoefficent = 2
printStatment= explorationRuns/5



##Run Game
for game in range(0,1):

  go_env = gym.make('Chess-v0')
  go_env.reset()
  startNode = Node(StepData(go_env._observation,0,0,0),0,0,None, None)
  gameTree = Tree(startNode)
  startNode = gameTree.get_start_node()
    
#TreeExploration
  actions = []

  currentNode = gameTree.get_start_node()
  done = False
  round = 0
  first = True
  while not done:
      t = time.process_time()
      for _ in range(0,explorationRuns):
        if(True):
          print("Tree Exploration Iteration: " + str(_))
          
        gameTree.expandingTree(currentNode,_ + game*explorationRuns,go_env, False)
      print("Time Used simulations: " + str(time.process_time() - t))
      blackAction = gameTree.findBestAction(currentNode ,round)
      actions.append(blackAction)
      first = False
      print("Blacks best Action is: ", end= "" )
      print(blackAction)
      state, reward, done, info = go_env.step(blackAction)
      print(go_env.render(mode='unicode'))
      currentNode = gameTree.findNodeBasedOnActions(actions)
      
      if(done):
          break

      #whiteAction =input("What should white play?: ")
      whiteAction = go_env.legal_moves[random.randrange(0,len(go_env.legal_moves))] 
      print("White round values:")
      #whiteAction = gameTree.findWorstAction(currentNode, round)
      print("White run values")
      actions.append(whiteAction)      
      print("White played: " + str(whiteAction))
      state, reward, done, info = go_env.step(whiteAction)
      print(go_env.render(mode='unicode'))
      
      if(done):
          break

      currentNode = gameTree.findNodeBasedOnActions(actions)
      
      print("actions So Far:" +str(actions))
      round +=1
        
  print("RESULT OF MATCH-------------------------------------")
  print(game)
  print(go_env.winning())
  print(go_env.reward())
  print("----------------------------------------------------")
  wins.append(go_env.winning())
  games.append(game+1)

plt.plot(games,wins)
plt.show()