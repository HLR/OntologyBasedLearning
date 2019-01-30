import os
from owlready2 import *
dash = '-' * 120

def print_concepts(concepts, concepts_type_name):
    spatial_concepts = list([currentConcept.name for currentConcept in concepts ])
    spatial_concepts.sort(key=str.lower)
    ont_batches = [spatial_concepts[i:i+4] for i in range(0, len(spatial_concepts), 4)]

    print(dash)
    tableTitle = concepts_type_name + " defined in SaulSpatial ontology"
    print('{:^120}'.format(tableTitle))
    print(dash)

    for batch in ont_batches :
        if len(batch) == 4:
            print(f'{batch[0]:30} {batch[1]:30} {batch[2]:30} {batch[3]:30}')
        elif len(batch) == 3:
            print(f'{batch[0]:30} {batch[1]:30} {batch[2]:30}')
        elif len(batch) == 2:
            print(f'{batch[0]:30} {batch[1]:30}')
        elif len(batch) == 1:
            print(f'{batch[0]:30}')
    print("\n")

onto_path.append("../ontology")

# Load Spatial Configuration ontology
mySaulSpatialOnto = get_ontology("http://ontology.ihmc.us/Spatial/SaulSpatialConfiguration.owl")
mySaulSpatialOnto.load()

# Get ontology classes defined in this ontology and print them in the table
spatial_classes = list(mySaulSpatialOnto.classes())
print_concepts(spatial_classes, "Classes")

# Get ontology properties defined in the ontology
spatial_properties = list(mySaulSpatialOnto.object_properties())
spatial_properties.extend(list(mySaulSpatialOnto.data_properties()))
print_concepts(spatial_properties, "Properties")

# Get individuals defined in this ontology and print them in the table
spatial_individuals = list(mySaulSpatialOnto.individuals())
print_concepts(spatial_individuals, "Individuals")

# Load SaulSpatial Examples
onto_path.append("../ontology/data")
data_ontologies = []

print("Found ontology data files: ",)        
for filename in os.listdir("../ontology/data") :
    if not filename.endswith(".owl") :
        continue
    
    filePath = os.path.join("../ontology/data", filename)
    print("   " + filename)
    
    f = open(filePath, "r")
    for line in f.readlines() :
        for item in line.split() :
            if item.startswith("xml:base") :
                indexesOfDoubleQuote = ( [pos for pos, char in enumerate(item) if char =='\"'] )
                ontology_name = item[indexesOfDoubleQuote[0]+1:indexesOfDoubleQuote[1]]
                data_ontologies.append(get_ontology(ontology_name).load())
print("\n")

graph = default_world.as_rdflib_graph()

# List defined spatialConfiguration and find their trajectors 
print('List defined spatialConfigurations:\n')
for instance in mySaulSpatialOnto.spatialConfiguration.instances(): 
    print("  ", instance)
    
    # get trajector for the current spatial configuration
    queryString1 = "SELECT ?tr WHERE {<" + instance.iri + "> <" +  mySaulSpatialOnto.hasTr.iri + "> ?tr .}"
#   print(queryString1)
    print("    Found trajector", list(graph.query_owlready(queryString1)))
    
    # get color for the current trajector
    queryString2 = "SELECT ?col WHERE {<" + instance.iri + "> <" +  mySaulSpatialOnto.hasTr.iri + "> ?tr . " + "?tr <" +  mySaulSpatialOnto.hasSpatialEntity.iri + "> ?se ."  + "?se <" +  mySaulSpatialOnto.col.iri + "> ?col .}" 
#   print(queryString2)
    print("    Found color of the trajector", list(graph.query_owlready(queryString2)), "\n")
    
    # get span text for the trajector
    queryString3 = "SELECT ?spanText WHERE {<" + instance.iri + "> <" +  mySaulSpatialOnto.hasTr.iri + "> ?tr . " + "?tr <" +  mySaulSpatialOnto.hasSpatialEntity.iri + "> ?se ."  + "?se <" +  mySaulSpatialOnto.hasSpan.iri + "> ?span ."  + "?span <" + mySaulSpatialOnto.hasSubText.iri + "> ?spanText .}"
#   print(queryString3)
    foundspans = list(graph.query_owlready(queryString3));
    print("    Found span text of the trajector", foundspans, "\n")
    
    # get motion indicator for the current spatial configuration
    queryString4 = "SELECT ?p WHERE {<" + instance.iri + "> <" +  mySaulSpatialOnto.hasM.iri + "> ?p .}"
#   print(queryString4)
    print("    Found motion indicator", list(graph.query_owlready(queryString4)), "\n")  