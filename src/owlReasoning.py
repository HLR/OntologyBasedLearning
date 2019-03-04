import os

from owlready2 import *

onto_path.append("../ontology") # the folder with the ontology

# Load Spatial Configuration ontology
mySaulSpatialOnto = get_ontology("http://ontology.ihmc.us/Spatial/SaulSpatialConfiguration.owl")
mySaulSpatialOnto.load()

cachedConsistencySearches = []
cachedCorrespondingResults = []

# Method checking if instance with all specified classes is consistent  with the ontolgy   
def testConsistencyOfInstance(*argv):
    
    appendedClasses = set()
    for arg in argv:  
        appendedClasses.add(arg)
        
    index = 0
    for s in cachedConsistencySearches:
        if s == appendedClasses:
            return cachedCorrespondingResults[index]
        else:
            index = index+1
           
    first = True 
    for appendedClass in appendedClasses:        
        if first:
            testSpatialEntity = appendedClass() # create instance of the class in the first karg
            first = False
        else:
            testSpatialEntity.is_a.append(appendedClass) # add subsequent karg as additional classes to the instance

    if first:  # if the method was called without correct classes  then no instance was created and nothing is consistent with ontology
        return True
    
    cachedConsistencySearches.append(appendedClasses)
    
    try:
        sync_reasoner() # call the reasoner to check if the new instance is consistent with the ontology definitions
    except OwlReadyInconsistentOntologyError as eInc:
        destroy_entity(testSpatialEntity)  # remove the instance  - clean the ontology for the next check
        cachedCorrespondingResults.append(False)
        return False                       # the instance is not consistent with the ontology
    
    destroy_entity(testSpatialEntity)      # remove the instance
    cachedCorrespondingResults.append(True)

    return True                            # the instance is consistent with the ontology - clean the ontology for the next check

print(testConsistencyOfInstance(mySaulSpatialOnto.tr, mySaulSpatialOnto.lm))
print(testConsistencyOfInstance(mySaulSpatialOnto.tr, mySaulSpatialOnto.sp))
print(testConsistencyOfInstance(mySaulSpatialOnto.tr, mySaulSpatialOnto.lm))