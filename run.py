import os, yaml, argparse, torch
from transformers import (set_seed, 
                          T5Config, 
                          T5TokenizerFast, 
                          T5ForConditionalGeneration, 
                          DataCollatorForSeq2Seq, 
                          TrainingArguments, 
                          Trainer)

from pynvml import (nvmlInit, 
                    nvmlDeviceGetHandleByIndex, 
                    nvmlDeviceGetMemoryInfo)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_model():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    return model


class Config(object):
    def __init__(self, args):
        self.mode = args.mode
        self.ckpt = f"{}.pt"

        self.n_epochs = 1
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.accumulation_steps = 4

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        if use_cuda:
            self.device_type = 'cuda'
        else:
            self.device_type = 'cpu'

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")


def memory_status():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    #print(f"GPU memory occupied: {info.used//1024**2} MB.")
    return info.used//1024**2


def main(args):
    set_seed(42)
    tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=512)

    if args.mode == 'train':
        pass
    elif args.mode == 'test':
        pass

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-amp', default=False, type=str2bool)
    parser.add_argument('-accumulate', default=False, type=str2bool)
    parser.add_argument('-checkpointing', default=False, type=str2bool)
    parser.add_argument('-optimizer', default='adam')
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test']
    assert args.model in ['bert', 't5', 'sparse']
    assert args.optimizer in ['adam', 'adafactor', '8bit']

    script = f'''
    * Mode: {args.mode}
    * Use AMP: {args.amp}
    * Use Gradient Accumulation: {args.accumulate}
    * Use Gradient Checkpointing: {args.checkpointing}
    * Optimizer: {args.optimizer}
    '''.replace(' ' * 4, '')
    
    print(script)
    main(args)