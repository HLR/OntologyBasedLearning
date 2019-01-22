from owlready2 import *

onto_path.append("../ontology")

mySaulSpatialOnto = get_ontology("http://ontology.ihmc.us/Spatial/SaulSpatial.owl")
mySaulSpatialOnto.load()

print('Classes defined in SaulSpatial ontology:')
print(list(mySaulSpatialOnto.classes()))
print("\n")

print('Properties defined in SaulSpatial ontology:')
print(list(mySaulSpatialOnto.object_properties()))
print("\n")

# Load SaulSpatial Examples
mySaulSpatialExamplesOnto = get_ontology("http://ontology.ihmc.us/Spatial/SaulSpatialExamples.rdf")
mySaulSpatialExamplesOnto.load()

graph = default_world.as_rdflib_graph()

# List defined spatialConfiguration and find their trajectors 
print('List defined spatialConfigurations:\n')
for instance in mySaulSpatialOnto.spatialConfiguration.instances(): 
    print("  ", instance)
    queryString1 = "SELECT ?p WHERE {<" + instance.iri + "> <" +  mySaulSpatialOnto.hasTr.iri + "> ?p .}"
#   print(queryString1)
    print("    Found trajector", list(graph.query_owlready(queryString1)))
    queryString2 = "SELECT ?p WHERE {<" + instance.iri + "> <" +  mySaulSpatialOnto.hasM.iri + "> ?p .}"
#   print(queryString2)
    print("    Found motion indicator", list(graph.query_owlready(queryString2)), "\n")