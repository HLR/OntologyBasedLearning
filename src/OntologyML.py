import os
import random
from pathlib import Path
from owlready2 import *

class OntologyMLModelCreator :
    'Class building graph (tree) of learners based on ontology'
    
    def __init__(self, name) :
        self.name = name
    
    def selectDataForConcept (self, concepts, data) :
        print("Collecting data for concepts", concepts)
        return data
    
    def buildGraph(self, classifierNode, data) :
        print("Building subgraph for", classifierNode.ontConcept)
        
        ontConceptSubclasses = classifierNode.ontConcept.subclasses(only_loaded = False, world = None)
        for subClass in ontConceptSubclasses :
            print("Current subclasse", subClass) # immediate subclasses without self
            
            # Create classfier for this subclass concept
            subClassClassifier = OntologyMLModelClassifier(subClass, data)
            
            # Add this classifier to the list in the current node
            classifierNode.subClassfier[subClass] = subClassClassifier
            
            # Extract data for the current subgraph
            subClassData = self.selectDataForConcept(subClass.descendants(include_self = True, only_loaded = False, world = None), data)
            
            # Init and train learning model
            subClassClassifier.learnModel(subClassData)
            
            # Recursively build subgraph for the current subclass
            self.buildGraph(subClassClassifier, subClassData)
    
    def build_model(self, ontology, data) :
        # Check if ontology path is correct
        ontologyPath = Path(os.path.normpath("../ontology"))
        if not os.path.isdir(ontologyPath.resolve()):
            print("Path to load ontology:", ontologyPath.resolve(), "does not exists")
            exit()

        onto_path.append(os.path.normpath("../ontology")) # the folder with the ontology

        # Load ontology
        myOnto = get_ontology(ontology)
        myOnto.load(only_local = False, fileobj = None, reload = False, reload_if_newer = False)
        
        # Get root class
        rootClass = None
        for cont_class in myOnto.classes():
            #print(cont_class)
            #print("ancestors", cont_class.ancestors(include_self = True)) # all parent
            #print("descendants", cont_class.descendants(include_self = True, only_loaded = False, world = None)) # all subclasses 
            
            #for subClass in cont_class.subclasses(only_loaded = False, world = None) :
            #    print("subclasse", subClass) # immediate subclasses without self
            
            for parent in cont_class.is_a : # immediate parent without self
                #print("    is a %s" %parent)
                if parent == owl.Thing :
                    print("Root")
                    rootClass = cont_class
                    break 
                
            if (rootClass != None) :
                #assuming single Root class
                break
        
        print("Found root class - ", rootClass)
        
        #print("ancestors", rootClass.ancestors(include_self = True))
        #print("descendants", rootClass.descendants(include_self = True, only_loaded = False, world = None))
        #for subClass in rootClass.subclasses(only_loaded = False, world = None) :
        #    print("subclasse", subClass)
        #
        #leafConcepts = []
        #for currentDescendant in descendants(include_self = True, only_loaded = False, world = None) :
        #   if bool(currentDescendant.subclasses(only_loaded = False, world = None) :
        #       leafConcepts.append(currentDescendant)
        #print("leafConcepts", leafConcepts)
        
        #subClasses = myOnto.search(subclass_of = rootClass)
        #print("subClasses", subClasses)

        # create new OntologyMLModelClassifier - sent to thread for init
        rootClassifier = OntologyMLModelClassifier(rootClass, data)
        
        # build graph
        self.buildGraph(rootClassifier, data)
        
        return rootClassifier
        
class OntologyMLModelClassifier :
    'Object containing graph of learners'
     
    def __init__(self, ontConcept, data) :
        self.ontConcept = ontConcept
        self.model = None
        self.subClassfier = dict()
        
    def learnModel(self, data):
        pass
    
    def shallowClassify(self, instance):
        # TODO integrate model
        return random.random() # Temporary
    
    def classify(self, instance) :
        print("Classifying -", instance, "- in the node for concept", self.ontConcept)
        
        # Get shallow classification from each child (using model build using combined data from all concepts below given child)
        shallowClassificationResults = dict()
        for currentConcept, currentClassfier in self.subClassfier.items() :
            shallowClassificationResults[currentConcept] = currentClassfier.shallowClassify(instance) # probability number
        
        # Decide which subnodes to follow 
        # right now select with the highest probability
        maxShallowClassificationResults = [key for m in [max(shallowClassificationResults.values())] for key,val in shallowClassificationResults.items() if val == m]
        
        classificationResults = dict()
        for currentMaxShallowClassificationResult in maxShallowClassificationResults: # concepts
           if bool(list(currentMaxShallowClassificationResult.subclasses(only_loaded = False, world = None))) : # check if empty
               # Not leaf - not empty
               selectedClassifierResults = self.subClassfier[currentMaxShallowClassificationResult].classify(instance)
               for selectedClassifierResult in selectedClassifierResults :
                   classificationResults[selectedClassifierResult[0]] = selectedClassifierResult[1]
           else:
               #Leaf
               classificationResults[currentMaxShallowClassificationResult] = shallowClassificationResults[currentMaxShallowClassificationResult] # Already found probability for this leaf
               
        # Decide which subnodes to follow 
        # right now select with the highest probability
        maxClassifiers = [key for m in [max(classificationResults.values())] for key,val in classificationResults.items() if val == m]
        
        myResult = [(key, classificationResults[key]) for key in maxClassifiers]
        return myResult
        
# --------- Testing

#shallowClassification = dict(concept1=.09, concept2=.08, concept3=.01, concept4=.09, concept5=.05, concept6=.01)
#maxList = [key for m in [max(shallowClassification.values())] for key,val in shallowClassification.items() if val == m]
#print(maxList)

myOntologyMLModelCreator = OntologyMLModelCreator("OntoNotes")
buildClassifier = myOntologyMLModelCreator.build_model("http://ontology.ihmc.us/ontonotes/ontonotes.owl", None)

print("\nClassify result", buildClassifier.classify("test"))