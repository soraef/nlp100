import xml.etree.ElementTree as ET

# 
# S:Turing, V:published, O: article
# 
# <dep type="nsubj">
#   <governor idx="6">published</governor>
#   <dependent idx="5">Turing</dependent>
# </dep>
# ...
# <dep type="dobj">
#   <governor idx="6">published</governor>
#   <dependent idx="8">article</dependent>
# </dep>
# 


tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

dependencies = root.findall('.//dependencies[@type="collapsed-dependencies"]')

for dependence in dependencies:
    # svo_dict[(id, v_name)] = {s: [(id, s_name), ...], o: [(id, o_name), ...]}
    svo_dict = {}
    for dep in dependence.iter("dep"):
        dep_type = dep.attrib["type"]
        governor  = dep.find("governor")
        dependent = dep.find("dependent")
        key = (governor.attrib["idx"], governor.text)
        if dep_type == "nsubj" or dep_type == "dobj":
            svo_dict.setdefault(key, {"s": [], "o": []})

        if dep_type == "nsubj":
            svo_dict[key]["s"].append((dependent.attrib["idx"], dependent.text))
        elif dep_type == "dobj":
            svo_dict[key]["o"].append((dependent.attrib["idx"], dependent.text))

    for key, value in svo_dict.items():
        v_name  = key[1]
        s_names = value["s"]
        o_names = value["o"]
        for s_name in s_names:
            for o_name in o_names:
                print(f"{s_name[1]}\t{v_name}\t{o_name[1]}") 
        

