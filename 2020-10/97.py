import subprocess

def train(name, encoder, decoder):
    subprocess.call(f"onmt_train -data data/demo-sub -save_model models/{name} --train_steps 20000 --world_size 3 --gpu_ranks 0 1 2 --encoder_type {encoder} --decoder_type {decoder} > /dev/null 2>&1", shell=True)

def translate(name):
    subprocess.call(f"onmt_translate -model models/{name}_step_20000.pt -src data/src-test-sub.txt -output pred.txt -replace_unk -verbose  > /dev/null 2>&1", shell=True)

def eval():
    subprocess.call("perl multi-bleu.perl data/tgt-test-sub.txt < pred.txt", shell=True)

params = [
    {"encoder": "rnn", "decoder": "rnn", "name": "test1"}, 
    {"encoder": "brnn", "decoder": "rnn", "name": "test2"}, 
    {"encoder": "cnn", "decoder": "cnn", "name": "test3"}, 
    {"encoder": "transformer", "decoder": "transformer", "name": "test4"}]

for param in params:
    print(f"{param}")
    train(param["name"], param["encoder"], param["decoder"])
    translate(param["name"])
    eval()

# {"encoder": "rnn", "decoder": "rnn", "name": "test1"}
# BLEU = 0.20, 20.6/4.0/2.7/1.8 (BP=0.044, ratio=0.243, hyp_len=408612, ref_len=1684675)
#
# {"encoder": "brnn", "decoder": "rnn", "name": "test2"}
# BLEU = 1.10, 16.5/3.3/1.9/1.1 (BP=0.334, ratio=0.477, hyp_len=803015, ref_len=1684675)
