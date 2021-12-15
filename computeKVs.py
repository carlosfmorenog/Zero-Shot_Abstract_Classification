def method4_1(target):
    '''This method follows the equation detailed in section 4.1 of the report to calculate k values'''
    import math
    print('\nCalculating k values...')
    ## Obtain the number of classes in label list and sort
    labelsIndexesUnique = list(set(target))
    labelsIndexesUnique.sort()
    ## For each class, count the number of instances and calculate ki
    k = []
    for label in labelsIndexesUnique:
        k.append(target.count(label))
    avgInst = sum(k)/len(k)
    k = [math.floor((ki/avgInst)+1) for ki in k]
    return k

def randomk(data, target):
    '''This method assigns random values to the k values from 1 to n, being n the number of samples in the class.'''
    import random
    k = []
    IndexesUnique = list(set(target))
    IndexesUnique.sort()
    for i, label in enumerate(IndexesUnique):
        ## Count the number of instances
        instances = 0
        for j in range(len(data)):
            if target[j]==label:
                instances+=1
        k.append(random.choice(range(instances)))
    return k

def randombinary(n_clusters, target):
    '''This method assigns a random value between either 1 (no clustering) or n_clusters (clustering) to the vector of k values.'''
    import random
    k = []
    for i in range(len(list(set(target)))):
        k.append(random.choice([[1,n_clusters]]))
    return k

def majority(data, target, n_clusters):
    '''This method finds the majority class and computes a vector of k values such that only the majority class is decomposed in 'n_clusters' clusters.'''
    IndexesUnique = list(set(target))
    IndexesUnique.sort()
    instances = [0]*len(IndexesUnique)
    k = [1]*len(IndexesUnique)
    ## Count the number of instances for each class
    for i, label in enumerate(IndexesUnique):
        for j in range(len(data)):
            if target[j]==label:
                instances[i]+=1
    ## Find the majority class
    majorityclass = instances.index(max(instances))
    k[majorityclass]=n_clusters
    return k