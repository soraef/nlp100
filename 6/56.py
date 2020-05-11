import xml.etree.ElementTree as ET

tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

# 文字変換表を作成する
# reconf_table[sentence_id][token_id] = 変換したい文字
reconf_table = {}
coreferences = root.findall(".//coreference/coreference")

for coreference in coreferences:
    representative_text = ""
    for mention in coreference.iter("mention"):
        # mentionタグにrepresentativeがついていたらそのテキストを保存
        if mention.attrib.get("representative"):
            representative_text = mention.find("text").text
            continue

        text        = mention.find("text").text          # 置換する文字
        sentence_id = int(mention.find("sentence").text) # textのあるsentenceのid
        start       = int(mention.find("start").text)    # 置換したい範囲の一番初めのtoken_id
        end         = int(mention.find("end").text)      # 置換したい範囲の一番最後のtoken_id

        # tableを初期化する
        # とりあえずsentence_idのstartからendまでを空白で置き換えるように設定
        for token_id in range(start, end):
            reconf_table[sentence_id] = {token_id: ""}
        
        # 変換したい文字を設定する
        reconf_table[sentence_id][start] = f"「{representative_text}({text})」"

# xmlをtextに置き換えていく
# 変換表に文字列があれば文字列を置き換える
def xml2text(root, reconf_table):
    sentences = root.findall(".//sentences/sentence")
    for sentence in sentences:
        sentence_id = int(sentence.attrib["id"])
        for token in sentence.iter("token"):
            token_id = int(token.attrib["id"])
            word = token.find("word").text

            # 変換表に変換すべき文字があればその文字を代入
            text_or_none = reconf_table.get(sentence_id, {}).get(token_id, None)
            
            if text_or_none is None:
                print(word, end=" ")
            elif text_or_none:
                print(text_or_none, end=" ")
                

xml2text(root, reconf_table)


# 
# 出力
# 
# Natural language processing From Wikipedia , the free encyclopedia Natural language processing -LRB- NLP -RRB- is 「the free encyclopedia Natural language processing -LRB- NLP -RRB-(a field of computer science)」 field of computer , artificial intelligence , and linguistics concerned with the interactions between computers and human -LRB- natural -RRB- languages . As such , NLP is related to the area of humani-computer interaction . Many challenges in NLP involve natural language understanding , that is , enabling 「computers(computers)」 to derive meaning from human or natural language input 
# , and others involve natural language generation . History The history of NLP generally starts in the 1950s , although work can be found from earlier 
# periods . In 1950 , Alan Turing published an article titled `` Computing Machinery and Intelligence '' which proposed what is now called the 「Alan Turing(Turing)」 test as a criterion of intelligence . The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English . The authors claimed that within three or five years , 「a solved problem(machine translation)」 would be a solved problem . 
# However , real progress was much slower , and after the ALPAC report in 1966 , which found that ten year long research had failed to fulfill the expectations , funding for machine translation was dramatically reduced . Little further research in 「a solved problem(machine translation)」 was conducted until the late 1980s , when the first statistical machine translation systems were developed . Some notably successful NLP systems developed in the 
# 1960s were SHRDLU , 「SHRDLU(a natural language system working in restricted `` blocks worlds '' with restricted vocabularies)」 natural language system working in restricted `` blocks worlds '' with restricted , and ELIZA , a simulation of a Rogerian psychotherapist , written by Joseph Weizenbaum between 1964 to 1966 . Using almost no information about human thought or emotion , ELIZA sometimes provided a startlingly human-like interaction . When the `` patient '' exceeded the very small knowledge base , ELIZA might provide a generic response , for example , responding to `` My head hurts '' 
# with `` Why do you say 「you(your)」 head hurts ? '' . During the 1970s many programmers began to write ` conceptual ontologies ' , which structured real-world information into computer-understandable data . Examples are MARGIE -LRB- Schank , 1975 -RRB- , SAM -LRB- Cullingford , 1978 -RRB- , PAM -LRB- Wilensky , 「1978(1978)」 -RRB- , TaleSpin -LRB- Meehan , 1976 -RRB- , QUALM -LRB- Lehnert , 1977 -RRB- , Politics -LRB- Carbonell , 1979 -RRB- , and Plot Units -LRB- Lehnert 1981 -RRB- . During this time , many chatterbots were written including PARRY , Racter , and Jabberwacky . Up to 「the late 1980s(the 1980s)」 , most NLP systems were based on complex sets of hand-written rules . Starting in 「the late 1980s(the late 1980s)」 late , however , there was a revolution in NLP with the introduction of machine learning algorithms for language processing . This was due to both the steady increase in computational power resulting from Moore 's Law and the gradual lessening of the dominance of Chomskyan theories of linguistics -LRB- e.g. transformational grammar -RRB- , whose theoretical underpinnings discouraged the sort of corpus linguistics that underlies the machine-learning approach 
# to 「the free encyclopedia Natural language processing -LRB- NLP -RRB-(language processing)」 . Some of the earliest-used machine learning 「machine learning algorithms for language processing(algorithms)」 , such as decision trees , produced systems of hard if-then rules similar to existing hand-written rules . However , Part of speech tagging introduced the use of Hidden Markov Models to NLP , and increasingly , research has focused on 「statistical models , which make soft , probabilistic decisions based on attaching real-valued weights to the features making up the input data(statistical models)」 , which make soft , probabilistic decisions based on attaching real-valued weights to the features making up the input data . The cache language models upon which many speech recognition systems now rely are 「The cache language models upon which many speech recognition systems now rely(examples of such statistical models)」 of such statistical . Such models are generally more robust when given unfamiliar input , especially 「as input(input that contains errors -LRB- as is very common for real-world data -RRB-)」 that contains errors -LRB- as is very common for real-world data , and produce more reliable results when integrated into a larger system comprising multiple subtasks . Many of the notable early successes occurred in the field of 「a solved problem(machine translation)」 , due especially to work at IBM Research , where successively more complicated statistical models were developed . These systems were able to take 「the advantage(advantage of existing multilingual textual corpora that had been produced by the Parliament of Canada and the European Union as a result of laws calling for the translation of all governmental proceedings into all official languages of the corresponding systems of government)」 of existing multilingual textual corpora that had been produced by the Parliament of Canada and the European Union as a result of laws calling for the translation of all governmental proceedings into all official languages of the corresponding systems of . However , most other systems depended on corpora specifically developed for the tasks implemented by these systems , which was -LRB- and often continues to be -RRB- a major limitation in the success of 「many speech recognition systems(these systems)」 . As a result , a great deal of research has gone into methods of more effectively learning from limited amounts of data . Recent research has increasingly focused on unsupervised and semi-supervised learning algorithms . Such algorithms are able to learn from 「data(data that has not been hand-annotated with the desired answers)」 that has not been hand-annotated with the desired , or using a combination of annotated and non-annotated data . Generally , this task is much more difficult than supervised learning , and typically produces less accurate results for a given amount of input data . However , there is an enormous amount of non-annotated data available -LRB- including , among other things , the entire content of the World Wide Web -RRB- , which can often make up for the inferior 
# results . NLP using machine learning Modern NLP algorithms are based on 「machine learning , especially statistical machine learning(machine learning)」 , especially statistical machine learning . The paradigm of 「machine learning , especially statistical machine learning(machine learning)」 is different from that of most prior attempts at language processing . Prior implementations of language-processing tasks typically involved the direct hand coding of large sets of 「hard if-then rules similar to existing hand-written rules(rules)」 . The machine-learning paradigm calls instead for using 
# general learning algorithms - often , although not always , grounded in statistical inference - to automatically learn such rules through the analysis of large corpora of typical real-world examples . A corpus -LRB- plural , `` corpora '' -RRB- is 「A corpus -LRB- plural , `` corpora '' -RRB-(a set 
# of documents -LRB- or sometimes , individual sentences -RRB- that have been hand-annotated with the correct values to be learned)」 set of documents -LRB- or sometimes , individual sentences -RRB- that have been hand-annotated with the correct values to be . Many different classes of 「machine learning Modern NLP algorithms(machine learning algorithms)」 learning have been applied to NLP tasks . 「machine learning Modern NLP algorithms(These algorithms)」 take as input a large set of `` features '' that are generated from the input data . Some of the earliest-used algorithms , such as 「decision trees(decision trees)」 , produced systems of hard if-then rules similar to the systems of hand-written rules that were then common . Increasingly 
# , however , research has focused on statistical models , which make soft , probabilistic decisions based on attaching 「real-valued weights(real-valued weights)」 to each input feature . Such models have the advantage that 「Some of the earliest-used algorithms , such as decision trees(they)」 can express the relative certainty of many different possible answers rather than only one , producing more reliable results when such a model is included 
# as a component of a larger system . Systems based on machine-learning algorithms have many advantages over hand-produced rules : The learning procedures used during 「machine learning , especially statistical machine learning(machine learning)」 automatically focus on the most common cases , whereas when writing rules by hand it is often not obvious at all where the effort should be directed . Automatic learning 「The learning procedures used during machine learning(procedures)」 can make use of statistical inference algorithms to produce models that are robust to unfamiliar input -LRB- e.g. containing words or structures that have not been seen before -RRB- and to erroneous input -LRB- e.g. with misspelled words or words accidentally omitted -RRB- . Generally , handling such input gracefully with hand-written rules -- or more generally , creating systems of 「hand-written rules(hand-written rules)」 that make soft decisions -- extremely difficult , error-prone and time-consuming . Systems based on automatically learning the rules can be made more accurate simply by supplying more input data . However , 「the systems(systems based on hand-written rules)」 based on hand-written can 
# only be made more accurate by increasing the complexity of the rules , which is a much more difficult task . In particular , there is a limit to the complexity of systems based on hand-crafted rules , beyond which the systems become more and more unmanageable . However , creating more data to input 
# to 「Systems based on machine-learning algorithms(machine-learning systems)」 simply requires a corresponding increase in the number of man-hours worked , generally without significant increases in the complexity of the annotation process . The subfield of NLP devoted to learning approaches is known as Natural Language Learning -LRB- NLL -RRB- and 「Natural Language Learning -LRB- NLL -RRB-(its)」 conference CoNLL and peak body SIGNLL are sponsored by ACL , recognizing also their links with Computational Linguistics and Language Acquisition . When the aims of computational language learning research is to understand more about human language acquisition , or psycholinguistics , NLL overlaps into the related field of Computational Psycholinguistics .