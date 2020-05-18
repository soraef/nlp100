with open("64_result.txt", "r") as f:
    data = f.readlines()

mode = "semantic"

semantic_count  = 0
syntactic_count = 0

semantic_correct_count  = 0
syntactic_correct_count = 0

for analogry in data:

    # 文法アナロジーになったらmodeを切り替える
    if analogry[0] == ":" and "gram" in analogry:
        mode = "syntactic"
    
    if analogry[0] == ":":
        continue

    true_word = analogry.split()[3]
    pred_word = analogry.split()[4]

    if mode == "semantic":
        semantic_count += 1
        if true_word == pred_word:
            semantic_correct_count += 1
    
    elif mode == "syntactic":
        syntactic_count += 1
        if true_word == pred_word:
            syntactic_correct_count += 1

print(f"semantic analogy accuracy: {semantic_correct_count / semantic_count}")
print(f"syntactic analogry accuracy: {syntactic_correct_count / syntactic_count}")

# semantic analogy accuracy: 0.7308602999210734
# syntactic analogry accuracy: 0.7400468384074942

    


