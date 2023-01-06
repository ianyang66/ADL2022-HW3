import json
import argparse
from tw_rouge import get_rouge


def main(args):
    refs, preds = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    with open(args.submission) as file:
        for line in file:
            line = json.loads(line)
            preds[line['id']] = line['title'].strip() + '\n'

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]
    jsonString = json.dumps(get_rouge(preds, refs), indent=2)
    print(jsonString)
    with open(args.result_file,'w') as file:
        file.write(jsonString)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-o', '--file_to_submission')
    parser.add_argument('-result', '--result_file')
    args = parser.parse_args()
    main(args)
