from owlready2 import *

onto_path.append("../ontology")

mySaulSpatialOnto = get_ontology("http://ontology.ihmc.us/Spatial/SaulSpatialConfiguration.owl")
mySaulSpatialOnto.load()

print('Classes defined in SaulSpatial ontology:')
print(list(mySaulSpatialOnto.classes()))
print("\n")

print('Properties defined in SaulSpatial ontology:')
print(list(mySaulSpatialOnto.object_properties()))
print("\n")

# Load SaulSpatial Examples
mySaulSpatialExamplesOnto = get_ontology("http://ontology.ihmc.us/Spatial/SaulSpatialExamples.owl")
mySaulSpatialExamplesOnto.load()

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
   