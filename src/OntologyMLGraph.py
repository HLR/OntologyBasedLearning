import os
from string import Template
from pathlib import Path
from owlready2 import *

class OntologyMLGraphCreator :
    'Class building graph based on ontology'
    
    # Templates for the elements of the Python graph
    graphHeaderTemplate = Template('\twith Graph(\'${graphName}\') as ${graphType}:\n')
    conceptTemplate = Template('\t$conceptName = Concept(name=\'${conceptName}\')\n')
    subclassTemplate = Template('\t${conceptName}.be(\'${superClassName}\')\n')
    relationTemplate = Template('\t${relationName}.be((${domanName}, ${rangeName}))\n')
    
    def __init__(self, name) :
        self.name = name
    
    def buildSubGraph(self, myOnto, graphRootClass, tabSize, graphFile) :
        print("\tBuilding subgraph for --- %s ----".expandtabs(2) %graphRootClass)
        
        # Collect all concept from this subgrapoh
        subGraphConcepts = []
        
        # Get graphName and graphType for teh subgraph from annotations of graphRootClass
        graphFile.write(self.graphHeaderTemplate.substitute(graphName=graphRootClass.graphName.first(), graphType=graphRootClass.graphType.first()).expandtabs(tabSize))

        # Increase tab for generated code
        tabSize+=tabSize
        
        # Add root concept to the graph
        graphFile.write(self.conceptTemplate.substitute(conceptName=graphRootClass._name).expandtabs(tabSize))
        for parent in graphRootClass.is_a : # immediate parent without self
            if parent != owl.Thing :
                graphFile.write(self.subclassTemplate.substitute(conceptName=graphRootClass._name, superClassName=parent._name).expandtabs(tabSize))

        subGraphConcepts.append(graphRootClass)
        
        # Add concepts in the subclass tree to the subgraph
        self.parseSubGraphOntology(graphRootClass, subGraphConcepts, tabSize, graphFile)
        
        # Add relations for every concepts found to this subgraph
        for subGraphConcept in subGraphConcepts:
            for ont_property in myOnto.object_properties() :
                domain = ont_property.get_domain().first() # Domain of the relation - assuming single domain   
                if ont_property.get_domain().first()._name == subGraphConcept._name : # if concept is a domain of this property
                    graphFile.write("\n")
                    graphFile.write(self.conceptTemplate.substitute(conceptName=ont_property._name).expandtabs(tabSize))

                    if ont_property.get_range().first() != None : # Check if property range is defined
                        graphFile.write(self.relationTemplate.substitute(relationName=ont_property._name, domanName = domain._name, rangeName=ont_property.get_range().first()._name).expandtabs(tabSize))

    def parseSubGraphOntology(self, ontConceptClass, subGraphConcepts, tabSize, graphFile):
        isLeaf = True

        ontConceptSubclasses = ontConceptClass.subclasses(only_loaded = False, world = None) # all the subclasses of the current concept
        for subClass in ontConceptSubclasses :
            print("\tCurrent subclasse".expandtabs(4), subClass) # immediate subclasses without self
            
            if subClass.graphType : # Check if this is a start of a new subgraph
                continue            # Skip it ands stop the parsing of this subtree
            
            isLeaf = False
            
            # Write concept and subclass relation to the subgraph
            graphFile.write(self.conceptTemplate.substitute(conceptName=subClass._name).expandtabs(tabSize))
            graphFile.write(self.subclassTemplate.substitute(conceptName=subClass._name, superClassName=ontConceptClass._name).expandtabs(tabSize))
            subGraphConcepts.append(subClass)
            
            # Recursively build subgraph for the current subclass
            self.parseSubGraphOntology(subClass, subGraphConcepts, tabSize , graphFile)
        
        if isLeaf :
            print("\tLeaf".expandtabs(6))
    
    def buildGraph(self, ontology, fileName="graph.py") :
        # Check if ontology path is correct
        ontologyPath = Path(os.path.normpath("../ontology/ML"))
        if not os.path.isdir(ontologyPath.resolve()):
            print("Path to load ontology:", ontologyPath.resolve(), "does not exists")
            exit()

        onto_path.append(ontologyPath) # the folder with the ontology
        #onto_path.append(os.path.normpath("../ontology/ML")) # the folder with the ontology

        # Load ontology
        myOnto = get_ontology(ontology)
        myOnto.load(only_local = False, fileobj = None, reload = False, reload_if_newer = False)
        print (myOnto.imported_ontologies)
        
        print("\n---- Building graph for ontology:", ontology)
        
        # Get root graph concepts
        rootGraphConcepts = set() # Set of found graph root concepts
        for cont_class in myOnto.classes():  
            if cont_class.graphType :          # Check if node annotated with graphType property
                rootGraphConcepts.add(cont_class)
         
        # Search imported ontology as well
        for currentOnt in myOnto.imported_ontologies :
            for cont_class in currentOnt.classes():  
                if cont_class.graphType :          
                    rootGraphConcepts.add(cont_class)
                    
        graphFile = open(fileName, "w")
        
        # Write Global Graph header
        graphFile.write(self.graphHeaderTemplate.substitute(graphName='global', graphType='graph').expandtabs(0));

        print("\nFound root graph concepts - \n")
        for rootConcept in rootGraphConcepts :
            # Build subgraph for each found graph root concept
            self.buildSubGraph(myOnto, rootConcept, 4, graphFile)
            graphFile.write("\n")

        graphFile.close()
        
        return graphFile.name
    
# --------- Testing

#-- Ontonotes
ontonotesOntologyMLGraphCreator = OntologyMLGraphCreator("Ontonotes")
ontonotesGraphFileName = ontonotesOntologyMLGraphCreator.buildGraph("http://ontology.ihmc.us/ML/ontonotesBridgeToMLGraph.owl", "OntonotesGraph.py") 

ontonotesGraphFile = open(ontonotesGraphFileName, 'r')
print("\nGraph build based on ontology - Python source code - %s\n\n" %ontonotesGraphFileName, ontonotesGraphFile.read())

#-- EMR
emrOntologyMLGraphCreator = OntologyMLGraphCreator("EMR")
emrGraphFileName = emrOntologyMLGraphCreator.buildGraph("http://ontology.ihmc.us/ML/EMR.owl", "EMRGraph.py")

emrGraphFile = open(emrGraphFileName, 'r')
print("\nGraph build based on ontology - Python source code - %s\n\n" %emrGraphFileName, emrGraphFile.read())
