# file path
import os
from pathlib import Path
from owlready2 import *

# check if ontology path is correct
ontologyPath = Path(os.path.normpath("../ontology"))
if not os.path.isdir(ontologyPath.resolve()):
    print("Path to load ontology:", ontologyPath.resolve(), "does not exists")
    exit()

onto_path.append(os.path.normpath("../ontology")) # the folder with the ontology

# Load Spatial Configuration ontology
mySaulSpatialOnto = get_ontology("http://ontology.ihmc.us/Spatial/SaulSpatialConfiguration.owl")
mySaulSpatialOnto.load(only_local = False, fileobj = None, reload = False, reload_if_newer = False)

cachedConsistencySearches = []
cachedCorrespondingResults = []

# Method checking if instance with all specified classes is consistent  with the ontolgy   
def testConsistencyOfInstance(*argv):

    # collect unique classes from kargs to append to the test instance
    appendedClasses = set()
    for arg in argv:  
        if (arg not in appendedClasses) and isinstance(arg, entity.ThingClass):  # test if karg is an ontology class
            appendedClasses.add(arg)
        
    # check if this search was already processed
    index = 0
    for s in cachedConsistencySearches:
        if s == appendedClasses:
            pass
            return cachedCorrespondingResults[index] # This search already answered - return cached result
        else:
            index = index+1
           
    cachedConsistencySearches.append(appendedClasses)

    # Build test instance with the classes appended to it
    noClassYetAppended = True 
    for appendedOntologyClass in appendedClasses:        
        if noClassYetAppended: # when noClassYetAppended class - then create the instance
            testOntologyInstance = appendedOntologyClass() # create instance of the class in the first karg
            noClassYetAppended = False
        else: # for the subsequent classes append the class to the instance
            testOntologyInstance.is_a.append(appendedOntologyClass) 

    if noClassYetAppended:  # if the method was called without correct classes then no instance was created and nothing is consistent with ontology
        cachedCorrespondingResults.append(True)
        return True
     
    # call reasoner to check if the ontology with the new instance is still consistent   
    try:
        sync_reasoner(x = None, debug = 0, keep_tmp_file = False) # call the reasoner to check if the new instance is consistent with the ontology definitions
    except OwlReadyInconsistentOntologyError as eInc:
        destroy_entity(testOntologyInstance)  # remove the instance  - clean the ontology for the next check
        cachedCorrespondingResults.append(False)
        return False                          # the instance is not consistent with the ontology
                                          
    destroy_entity(testOntologyInstance)      # remove the instance
    cachedCorrespondingResults.append(True)

    return True                            # the instance is consistent with the ontology - clean the ontology for the next check

#print(testConsistencyOfInstance(mySaulSpatialOnto.tr, mySaulSpatialOnto.tr))

#print(testConsistencyOfInstance(mySaulSpatialOnto.tr, mySaulSpatialOnto.lm))
#print(testConsistencyOfInstance(mySaulSpatialOnto.tr, mySaulSpatialOnto.sp))
#print(testConsistencyOfInstance(mySaulSpatialOnto.tr, mySaulSpatialOnto.lm))