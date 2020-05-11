import xml.etree.ElementTree as ET
import pprint

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
    # key:   述語のidと名前のタプル
    # value: 述語に依存する主語(s)と目的語(o)の辞書; リストに語のidと名前が入っている
    svo_dict = {}
    
    for dep in dependence.iter("dep"):
        dep_type = dep.attrib["type"]
        governor  = dep.find("governor")
        dependent = dep.find("dependent")
        key = (governor.attrib["idx"], governor.text) # 述語のkey

        # dep_typeが述語-主語, 述語-目的語の関係だった場合辞書を初期化(keyがない場合のみ)
        if dep_type == "nsubj" or dep_type == "dobj":
            svo_dict.setdefault(key, {"s": [], "o": []}) 

        if dep_type == "nsubj":
            svo_dict[key]["s"].append((dependent.attrib["idx"], dependent.text))
        elif dep_type == "dobj":
            svo_dict[key]["o"].append((dependent.attrib["idx"], dependent.text))

    for v_id_name, so_dict in svo_dict.items():
        v_name  = v_id_name[1]
        s_id_name_list = so_dict["s"]
        o_id_name_list = so_dict["o"]

        # 主語述語目的語の組み合わせを全通り出力
        # 今回はs_id_name_listやo_id_name_listが複数あることはなかった
        for s_id_name in s_id_name_list:
            for o_name in o_id_name_list:
                print(f"{s_id_name[1]}\t{v_name}\t{o_name[1]}") 

        
# 
# 出力
# 
# understanding   enabling        computers
# others  involve generation
# Turing  published       article
# experiment      involved        translation
# ELIZA   provided        interaction
# patient exceeded        base
# ELIZA   provide response
# which   structured      information
# underpinnings   discouraged     sort
# that    underlies       approach
# Some    produced        systems
# which   make    decisions
# systems rely    which
# that    contains        errors
# implementations involved        coding
# algorithms      take    set
# Some    produced        systems
# which   make    decisions
# models  have    advantage
# they    express certainty
# Systems have    advantages
# Automatic       make    use
# that    make    decisions