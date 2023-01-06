from argparse import ArgumentParser, Namespace
from datasets import Dataset, load_dataset
from dataset import MT5Dataset
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import(
    AutoTokenizer, 
    MT5ForConditionalGeneration
    )
import os
from tqdm import tqdm
from accelerate import Accelerator
from tw_rouge import get_rouge

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(args):
    # prepare acclerator device
    accelerator = Accelerator()
    accelerator = Accelerator(fp16=True)
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    model.to(device)

    dataset = load_dataset('json', data_files={'dev': args.test_file})
    dev_dataset = MT5Dataset(tokenizer=tokenizer, dataset=dataset['dev'], mode='dev', 
                            input_length=args.max_source_length, output_length=args.max_target_length)

    val_loader = DataLoader(dataset=dev_dataset, batch_size=args.test_batch_size, shuffle=False)

    model, val_loader = accelerator.prepare(model, val_loader)

    model.load_state_dict(torch.load(args.ckpt_path))
    # validate model
    pred, titles, ids = [], [], []
    model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader):
            texts = data['source_ids'].squeeze().to(device)
            masks = data['source_mask'].to(device)
            title = data['target']
            out = model.generate(
                input_ids=texts, 
                attention_mask=masks, 
                max_length=args.max_target_length, 
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature
            )
            output = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for a, b, label_id in zip(output, title, data['id']):
                if a:
                    pred.append(a)
                    titles.append(b)
                    ids.append(label_id)

    eval_pred = get_rouge(pred, titles)
    print(eval_pred['rouge-1']['f'], eval_pred['rouge-2']['f'], eval_pred['rouge-l']['f'])
    with open(args.output_json, 'w+') as fout:
        for p, label_id in zip(pred, ids):
            print({"title": p, "id": label_id}, file=fout)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="ckpt/2_RL.ckpt")
    parser.add_argument("--test_file", type=str, default="data/public.jsonl")
    parser.add_argument("--output_json", type=str, default="submission3-RL.jsonl")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--max_source_length", type=int, default=384)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_beams", type=int, default=6)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)