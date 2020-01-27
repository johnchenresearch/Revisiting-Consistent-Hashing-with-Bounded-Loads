#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:14:12 2020

@author: johnchen
"""
import numpy as np
import random
import collections
import mmh3

class RandomJumpCHSimple():
    """
    Simple implementation of Random Jump Consistent Hashing (RJCH). This is a faster way of
    producing server load variance, expected number of full bins and other
    properties, since RJCH assign objects with equal probability to the non-full
    bins.
    """

    def __init__(self, servers=1000, duplicates=1, objects=10000, epsilon=0.3):
        """
        servers - How many unique servers.
        duplicates - how many virtual copies of each server.
        objects - how many objects to put in.
        epsilon - (1 + epsilon) * objects / servers is the load cap for each server.
        """
        self.serversCount = servers
        self.duplicatesCount = duplicates
        self.objectsCount = objects
        self.epsilon = epsilon
        self.loadCap = int(np.ceil((1 + epsilon) * objects / servers))
    
    def start(self):
        """
        Initializes
        """
        self.firstFull = np.float('inf')
        
        self.servers = {i:0 for i in range(self.serversCount)} # Mapping from serverid: number of objects
        self.fullFlag = {i:False for i in range(self.serversCount)}
        
        randNums = np.random.randint(0, self.serversCount, size=self.objectsCount)
        
        counter = 0
        for num in randNums:
            counter += 1
            # Make sure not full.
            while self.fullFlag[num]:
                num = np.random.randint(0, self.serversCount)
                
            self.servers[num] += 1
            if self.servers[num] == self.loadCap:
                self.fullFlag[num] = True
                self.firstFull = min(counter, self.firstFull)
            
        
    def variance(self):
        """
        Returns current variance.
        """
        return np.var(list(self.servers.values()))
        
    def serversToTry(self):
        """
        Simulates throwing another object in and seeing how many
        servers it needs to try before finding a non-full one.
        """
        num = np.random.randint(0, self.serversCount)
        counter = 1
        while self.fullFlag[num]:
            num = np.random.randint(0, self.serversCount)
            counter += 1
        return counter
        
        
    def pctOfFullBins(self):
        """
        Returns the pct of full bins.
        """
        return sum(self.fullFlag.values()) / len(self.fullFlag)
        
    def objectsTillFirstFull(self):
        """
        Returns the objects needed until the first bin is full
        """
        return self.firstFull
    
class RandomJumpConsistentHashing():
    """
    One implementation of Random Jump Consistent Hashing (RJCH).
    """
    def __init__(self, servers=1000, duplicates=1, objects=10000, epsilon=0.3):
        """
        servers - How many unique servers.
        duplicates - how many virtual copies of each server.
        objects - how many objects to put in.
        epsilon - (1 + epsilon) * objects / servers is the load cap for each server.
        """
        self.serversCount = servers
        self.totalServersAndPastCount = servers
        self.duplicatesCount = duplicates
        self.objectsCount = objects
        self.totalObjectsAndPastCount = objects
        self.epsilon = epsilon
        self.loadCap = int(np.ceil((1 + epsilon) * objects / servers))
        self.totalFull = 0
    
    def start(self):
        """
        Initializes and assigns objects to servers according to init values.
        """
        self.servers = {i:0 for i in range(self.serversCount)} # Mapping from serverid: number of objects
        self.fullFlag = {i:False for i in range(self.serversCount)}
        self.serversToIdx = {} # Mapping of servers to the index in the long array.
        
        # Put the servers in a long array.
        self.serversArray = np.array([-1 for i in range(2**20)])#range(self.serversCount * self.duplicatesCount * 100)])#
        self.serversArrayLen = len(self.serversArray)
        
        # Assign duplicates.
        for i in self.servers:
            dups = []
            for j in range(self.duplicatesCount):
                cur = np.random.randint(0, self.serversArrayLen)
                while self.serversArray[cur] != -1:
                    cur = np.random.randint(0, self.serversArrayLen)
                self.serversArray[cur] = i
                dups.append(cur)
            self.serversToIdx[i] = dups
        
        # Used to track what is contained in the server.
        self.serversContains = {i: set() for i in range(self.serversCount)}
        
        # Used to track all previous tries.
        self.objectsHistory = {i: [] for i in range(self.objectsCount)}
        
        # Where the object is now.
        self.objectsCurrent = {i: 0.5 for i in range(self.objectsCount)}
       

        for i in range(self.objectsCount):
            # In practice refer to the assignObjectWallTime method for randomization.
            num = np.random.randint(0, self.serversArrayLen)
            self.objectsHistory[i].append(num)
            
            # Make sure not full.
            while (self.serversArray[num] == -1) or (self.fullFlag[self.serversArray[num]]):
                num = np.random.randint(0, self.serversArrayLen)
                self.objectsHistory[i].append(num)
                
            curServer = self.serversArray[num]
            self.serversContains[curServer].add(i)
            self.servers[curServer] += 1
            if self.servers[curServer] == self.loadCap:
                self.fullFlag[curServer] = True
                self.totalFull += 1

            self.objectsCurrent[i] = num
            
    def variance(self):
        """
        Returns current server load variance.
        """
        return np.var(list(self.servers.values()))
        
    def assignObjectServersTried(self):
        """
        Simulates throwing another object in and seeing how many
        servers it needs to try before finding a non-full one.
        """
        num = np.random.randint(0, self.serversArrayLen)
        counter = 1
        while self.serversArray[num] == -1 or self.fullFlag[self.serversArray[num]]:
            num = np.random.randint(0, self.serversArrayLen)
            if self.serversArray[num] != -1 and self.fullFlag[self.serversArray[num]]:
                counter += 1
        return counter
        
    def assignObjectTotalSteps(self):
        """
        Simulates throwing another object in and seeing how many
        total steps it needs to try before finding a non-full one.
        """
        num = np.random.randint(0, self.serversArrayLen)
        counter = 1
        while self.serversArray[num] == -1 or self.fullFlag[self.serversArray[num]]:
            num = np.random.randint(0, self.serversArrayLen)
            counter += 1
        return counter
        
    def pctOfFullBins(self):
        """
        Returns the pct of full bins.
        """
        return sum(self.fullFlag.values()) / len(self.fullFlag)
    

    def assignObjectWallTime(self):
        """
        Tries to add an object and returns the wall time.
        """
        t = time.time()
        inputStr = str(np.random.random())
        num = mmh3.hash64(inputStr)
        num1 = num[0] & 4294967295
        num2 = num[0] >> 32
        num3 = num[1] & 4294967295
        num4 = num[1] >> 32
        
        num1 = num1 >> 12
        num2 = num2 >> 12
        num3 = num3 >> 12
        num4 = num4 >> 12
        counter = 0

        while (self.serversArray[num1] == -1 or self.fullFlag[self.serversArray[num1]]) and (self.serversArray[num2] == -1 or self.fullFlag[self.serversArray[num2]]) and (self.serversArray[num3] == -1 or self.fullFlag[self.serversArray[num3]]) and (self.serversArray[num4] == -1 or self.fullFlag[self.serversArray[num4]]):
            counter += 1
            
            num = mmh3.hash64(inputStr + hex(counter))
            num1 = num[0] & 1048575 # Always positive
            num2 = num[0] >> 44
            num3 = num[1] & 1048575 # Always positive
            num4 = num[1] >> 44          
        
        return time.time() - t
             
    def addOneObject(self):
        
        num = np.random.randint(0, self.serversArrayLen)
        
        # Update object history.
        self.objectsHistory[self.totalObjectsAndPastCount] = [num]
        
        # If no bin or full bin.
        while self.serversArray[num] == -1 or self.fullFlag[self.serversArray[num]]:
            num = np.random.randint(0, self.serversArrayLen)
            self.objectsHistory[self.totalObjectsAndPastCount].append(num)

        curServer = self.serversArray[num]
        
        # Found non-full bin
        self.servers[curServer] += 1
        if self.servers[curServer] == self.loadCap:
            self.fullFlag[curServer] = True
            self.totalFull += 1
            
        self.serversContains[curServer].add(self.totalObjectsAndPastCount)
        self.objectsCurrent[self.totalObjectsAndPastCount] = num
        self.objectsCount += 1
        self.totalObjectsAndPastCount += 1
        
    def removeOneObject(self):
                
        # Get the object that this count refers to.
        objectNum = np.random.choice(list(self.objectsCurrent.keys()))

        # Get the server that object is in.
        serverIdx = self.objectsCurrent[objectNum]

        serverKey = self.serversArray[serverIdx]
        
        # Delete the object from the server total.
        self.serversContains[serverKey].remove(objectNum)
        
        self.servers[serverKey] -= 1
        
        self.objectsCount -= 1
        self.objectsCurrent.pop(objectNum)
        self.objectsHistory.pop(objectNum)
        
        # Refill bin as much as possible. also change self.totalFull
        if self.fullFlag[serverKey]:
            self.fullFlag[serverKey] = False
            self.totalFull -= 1
            
            self.fillBinOne(serverIdx)
      

    
    def fillBinOne(self, serverIdx):
        """
        Fills the bin if it has one missing.
        """
        serverKey = self.serversArray[serverIdx]

        if self.servers[serverKey] != (self.loadCap - 1):
            return

        dups = [i for i in self.serversToIdx[serverKey]]
        np.random.shuffle(dups)
        
        lstObjects = list(self.objectsHistory.keys())
        np.random.shuffle(lstObjects)
        # Now we need to try and refill it.
        #for i in self.objectsHistory:
        for i in lstObjects:
            for serverIdx in dups: #new
                if (serverIdx in self.objectsHistory[i]) and \
                (self.serversArray[self.objectsCurrent[i]] != serverKey):
    
                    self.serversContains[serverKey].add(i)
                    self.servers[serverKey] += 1
    
                    self.serversContains[self.serversArray[self.objectsCurrent[i]]].remove(i)
                    self.servers[self.serversArray[self.objectsCurrent[i]]] -= 1
                    if self.fullFlag[self.serversArray[self.objectsCurrent[i]]] == True:
                        self.fullFlag[self.serversArray[self.objectsCurrent[i]]] = False
                        self.totalFull -= 1
    
                    idx = self.objectsHistory[i].index(serverIdx)
                    self.objectsHistory[i] = self.objectsHistory[i][:idx + 1]
                    
                    binUnfilled = self.objectsCurrent[i]
                    
                    self.objectsCurrent[i] = serverIdx
    
                    self.fillBinOne(binUnfilled)
                    
                    # We can stop if we've already filled the current bin. This condition can only happen within an if.
                    if self.servers[serverKey] == self.loadCap and self.fullFlag[serverKey] == False:
                        self.totalFull += 1
                        self.fullFlag[serverKey] = True
                        return
        

    
    

