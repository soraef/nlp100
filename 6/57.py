import graphviz
import xml.etree.ElementTree as ET

def set_up_dot(filename, datas_dir):

    dot = graphviz.Digraph(
                comment='かかり受け木',
                filename=filename, # DOT言語ファイルのファイル名 (これがグラフ画像のファイル名にも使われる)
                directory=datas_dir, # DOT言語ファイルと画像を保存するフォルダ
                format='png', # グラフの保存形式
                engine='dot',
                )

    fontname = 'MS Gothic'
    dot.attr('graph', fontname=fontname)
    dot.attr('node', fontname=fontname, shape='box', color='blue', style='rounded')
    dot.attr('edge', fontname=fontname, penwidth='1.5', color='gray')

    return dot

dot = set_up_dot("dot_57", "./")

tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

dependencies = root.findall('.//dependencies[@type="collapsed-dependencies"]')

for dependence in dependencies[1:2]:
    for dep in dependence.iter("dep"):
        governor  = dep.find("governor")
        dependent = dep.find("dependent")

        # dot.node(識別子, 表示名)
        dot.node(governor.attrib["idx"], governor.text)
        dot.node(dependent.attrib["idx"], dependent.text)

        dot.edge(governor.attrib["idx"], dependent.attrib["idx"])

dot.render()



