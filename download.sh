mkdir ./best-summarization
mkdir ./ckpt

wget https://www.dropbox.com/s/3o0omdunwspywc0/config.json?dl=1 -O ./best-summarization/config.json
wget -t 3 -T 30 -c https://www.dropbox.com/s/k3knytzctz80ul7/pytorch_model.bin?dl=1 -O ./best-summarization/pytorch_model.bin
wget -t 3 -T 30 -c https://www.dropbox.com/s/k3knytzctz80ul7/pytorch_model.bin?dl=1 -O ./best-summarization/pytorch_model.bin
wget -t 3 -T 30 -c https://www.dropbox.com/s/k3knytzctz80ul7/pytorch_model.bin?dl=1 -O ./best-summarization/pytorch_model.bin
wget https://www.dropbox.com/s/2s9yla00bf0ci14/special_tokens_map.json?dl=1 -O ./best-summarization/special_tokens_map.json
wget https://www.dropbox.com/s/ltzhjb2zpt7r1dr/spiece.model?dl=1 -O ./best-summarization/spiece.model
wget https://www.dropbox.com/s/15jjwtjzxxneb42/tokenizer.json?dl=1 -O ./best-summarization/tokenizer.json
wget https://www.dropbox.com/s/t3j25fyqtrtp2ob/tokenizer_config.json?dl=1 -O ./best-summarization/tokenizer_config.json
wget https://www.dropbox.com/s/vsvga2lncaoc0i5/training_args.bin?dl=1 -O ./best-summarization/training_args.bin
