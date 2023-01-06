# Homework 3 - NTU ADL 2022 FALL

## Reproduce testing process

```shell
bash download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

example:

```shell
bash download.sh
bash ./run.sh ./data/public.jsonl ./data/best-sumission.jsonl
```

<br>



## Reproduce training process

### Training & Evaluation

```shell
bash train.sh  /path/to/train.jsonl /path/to/valid.jsonl /path/to/model_output_directory
```

example:

```shell
bash train.sh ./data/train.jsonl ./data/public.jsonl ./best-summarization
```



### Training & Evaluation with my RL algorithm.

#### Training

```shell
bash train_RL.sh /path/to/train.jsonl /path/to/valid.jsonl /path/to/model_output_directory
```

example:

```shell
bash train_RL.sh ./data/train.json ./data/public.jsonl ./ckpt
```



####　Testing

```shell
bash infer_RL.sh /path/to/test.jsonl /path/to/output.jsonl
```

example:

```shell
bash infer_RL.sh ./data/public.jsonl ./data/RL_public_predict.jsonl
```



## Reference

[moooooser99/ADL22-HW3](https://github.com/moooooser999/ADL22-HW3)

[Transformers Summarization](https://github.com/huggingface/transformers/tree/t5-fp16-no-nans/examples/pytorch/summarization)