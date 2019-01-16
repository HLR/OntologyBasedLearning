from owlready2 import *

onto_path.append("../ontology")
myOnto = get_ontology("http://ontology.ihmc.us/Spatial/SaulSpatial.owl")
myOnto.load()

print(list(myOnto.classes()))
print(list(myOnto.object_properties()))

