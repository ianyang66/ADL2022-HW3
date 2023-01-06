from argparse import ArgumentParser, Namespace
from datasets import Dataset, load_dataset
from dataset import MT5Dataset
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer, 
    MT5ForConditionalGeneration, 
    Adafactor
)
import json
from tqdm import trange, tqdm
from tw_rouge import get_rouge
import os
from accelerate import Accelerator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(args):
    # prepare acclerator device
    accelerator = Accelerator()
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    model.to(device)

    dataset = load_dataset('json', data_files={'train': args.train_file})
    train_dataset = MT5Dataset(tokenizer=tokenizer, dataset=dataset['train'], mode='train', 
                            input_length=args.max_source_length, output_length=args.max_target_length)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    
    optimizer = Adafactor(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, relative_step=False)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    step = 0
    #model.load_state_dict(torch.load('./ckpt/0_RL.ckpt'))
    for epoch in trange(args.num_train_epoch):
        # train model
        model.train()
        train_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            step += 1
            source_ids, target_ids = data['source_ids'].to(device), data['target_ids'].to(device)
            # masks, title_masks = data['source_mask'].to(device), data['target_mask'].to(device)
            titles = data['target']

            out = model.generate(input_ids=source_ids, max_length=args.max_target_length)
            outputs = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            for idx, output in enumerate(outputs):
                outputs[idx] += '\n'

            rouges = get_rouge(outputs, titles)
            score = rouges['rouge-l']['f']/0.22
            
            loss = model(input_ids=source_ids, labels=target_ids).loss
            loss *= score

            loss /= args.gradient_accu_step
            # loss.backward()
            accelerator.backward(loss)
            train_loss += loss.item()
            if step % args.gradient_accu_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            
        print(f"\nTraining loss: {train_loss / len(train_loader) / args.train_batch_size}\n", flush=True)
        train_loss = 0

        torch.save(model.state_dict(), f"./ckpt/{epoch}_RL.ckpt")

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="ckpt/best_RL.ckpt")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")

    parser.add_argument("--num_train_epoch", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=1)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--gradient_accu_step", type=int, default=32)
    parser.add_argument("--max_source_length", type=int, default=384)
    parser.add_argument("--max_target_length", type=int, default=64)

    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)