#from grasp import Grasp
from place import Place
#grasp = Grasp(None, None, {"multimodal_similarity_threshold": 0.2})
place = Place(None, None, {"multimodal_similarity_threshold": 0.2})
#print(grasp("an apple"))
print(place("a desk"))