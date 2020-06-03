import subprocess

def translate(size):
    subprocess.call(f"onmt_translate -model demo-model_step_95000.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose -max_length {size}", shell=True)

def eval():
    subprocess.call("perl multi-bleu.perl tgt-test.txt < pred.txt", shell=True)

for i in range(1, 102, 20):
    print(f"beam_max_len: {i}")
    translate(i)
    eval()




# beam_max_len: 1
# BLEU = 0.00, 16.0/0.0/0.0/0.0 (BP=0.000, ratio=0.041, hyp_len=300, ref_len=7343)
# beam_max_len: 21
# BLEU = 0.13, 21.4/5.8/2.3/0.8 (BP=0.034, ratio=0.228, hyp_len=1673, ref_len=7343)
# beam_max_len: 41
# BLEU = 0.18, 19.2/5.1/2.0/0.6 (BP=0.053, ratio=0.254, hyp_len=1868, ref_len=7343)
# beam_max_len: 61
# BLEU = 0.20, 18.2/4.8/1.8/0.6 (BP=0.066, ratio=0.269, hyp_len=1975, ref_len=7343)
# beam_max_len: 81
# BLEU = 0.23, 17.2/4.5/1.7/0.5 (BP=0.082, ratio=0.285, hyp_len=2095, ref_len=7343)
# beam_max_len: 101
# BLEU = 0.25, 16.6/4.3/1.6/0.5 (BP=0.091, ratio=0.295, hyp_len=2163, ref_len=7343)