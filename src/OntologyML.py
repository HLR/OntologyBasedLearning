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
            print("\nCurrent subclasse", subClass) # immediate subclasses without self
            
            # Create classfier for this subclass concept
            subClassClassifier = OntologyMLModelClassifier(subClass, data)
            
            # Add this classifier to the list in the current node
            classifierNode.subClassfiers[subClass] = subClassClassifier
            
            # Extract data for the current subgraph
            subClassData = self.selectDataForConcept(subClass.descendants(include_self = True, only_loaded = False, world = None), data)
            
            # Init and train learning model
            subClassClassifier.learnModel(subClassData)
            
            # Recursively build subgraph for the current subclass
            self.buildGraph(subClassClassifier, subClassData)
        
        if not bool(classifierNode.subClassfiers) :
            print("Leaf")
    
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
        
        print("\n---- Building graph for ontology:", ontology)
        # Get root class
        rootClass = None
        for cont_class in myOnto.classes():
            for parent in cont_class.is_a : # immediate parent without self
                if parent == owl.Thing :
                    rootClass = cont_class
                    break 
                
            if (rootClass != None) :
                #assuming single Root class
                break
        
        print("\nFound root class - ", rootClass, "\n")

        # Create new OntologyMLModelClassifier - TODO sent to thread for init
        rootClassifier = OntologyMLModelClassifier(rootClass, data)
        
        # Build graph
        self.buildGraph(rootClassifier, data)
        
        return rootClassifier
        
class OntologyMLModelClassifier :
    'Object containing graph of learners'
     
    def __init__(self, ontConcept, data) :
        self.ontConcept = ontConcept
        self.model = None
        self.subClassfiers = dict()
        
    def learnModel(self, data):
        pass # TODO
    
    def shallowClassify(self, instance):
        # TODO integrate model
        return random.random() # Temporary
    
    def selectNodesForSubGraphClassification(self, shallowClassificationResults):
        # Right now select with the highest probability
        selectedNodes = [key for m in [max(shallowClassificationResults.values())] for key,val in shallowClassificationResults.items() if val == m] 
        
        return selectedNodes # concepts associated with node
    
    def decideOnClassificationResult(self, classificationResults):
         # Right now select with the highest probability
        selectedClassificationResults = [key for m in [max(classificationResults.values())] for key,val in classificationResults.items() if val == m] 
        
        return selectedClassificationResults # concepts associated with result
        
    def classify(self, instance) :
        print("Classifying -", instance, "- in the node for concept", self.ontConcept)
        
        # Get shallow classification from each child (using model build using combined data from all concepts below given child)
        shallowClassificationResults = dict()
        for currentConcept, currentClassfier in self.subClassfiers.items() :
            shallowClassificationResults[currentConcept] = currentClassfier.shallowClassify(instance) # probability number
        
        # Decide which subnodes to follow 
        selectedNodes = self.selectNodesForSubGraphClassification(shallowClassificationResults)
        
        classificationResults = dict()
        for selectedNode in selectedNodes : 
           if bool(list(selectedNode.subclasses(only_loaded = False, world = None))) : # check if empty
               # Not empty - not leaf concept
               selectedClassifierResults = self.subClassfiers[selectedNode].classify(instance) # run subgraph classification
               
               # Add found classification results to classificationResults
               for selectedClassifierResult in selectedClassifierResults :
                   classificationResults[selectedClassifierResult[0]] = selectedClassifierResult[1]
           else:
               # Leaf concept
               # Copy previous result to classificationResults
               classificationResults[selectedNode] = shallowClassificationResults[selectedNode] # Already found probability for this leaf
               
        # Decide which results to select 
        selectedResults = self.decideOnClassificationResult(classificationResults)
        
        # Build list of tuples with concept and probability
        myResult = [(key, classificationResults[key]) for key in selectedResults]
        
        return myResult
        
# --------- Testing

myOntologyMLModelCreator = OntologyMLModelCreator("OntoNotes")
ontoNotesClassifier = myOntologyMLModelCreator.build_model("http://ontology.ihmc.us/ontonotes/ontonotes.owl", None)

print("\n-- Test Classification algorithm with dummy instance - traversing graph of learners")
print("\nClassification result - ", ontoNotesClassifier.classify("test"))