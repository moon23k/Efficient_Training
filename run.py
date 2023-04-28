import os, yaml, argparse, torch
from datasets import Dataset
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



class Config(object):
    def __init__(self, args):
        self.mode = args.mode
        self.strategy = args.strategy
        self.mname = 't5-small'

        self.n_epochs = 1
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.ckpt = f"ckpt/strategy_{self.strategy}"
        self.output_dir = f"outputs/strategy_{strategy}/output"
        self.logging_dir = f"outputs/strategy_{strategy}/logging"
        
        self.fp_16= True if self.strategy > 0 else False
        self.fp16_opt_level="02" if self.fp_16 else "01"
        self.gradient_accumulation_steps= 4 if self.strategy > 1 else 1
        self.gradient_checkpointing = True if self.strategy > 2 else False
        self.optim = 'adafactor' if self.strategy > 3 else 'adamw_torch'

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_model(config):
    if config.mode == 'train':
        model = T5ForConditionalGeneration.from_pretrained(config.mname)
    else:
        model_config = T5Config.from_pretrained(config.mname)
        model = T5ForConditionalGeneration(model_config)        
        ckpt = config.ckpt
        assert os.path.exists(ckpt)
        model_state = torch.load(ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
    
    return model.to(config.device)




def print_memory():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")



def set_training_args(config):
    training_args_dict = {'group_by_length': True,
                          'save_strategy': 'epoch',
                          'logging_strategy': 'epoch',
                          'evaluation_strategy': 'epoch',

                          'disable_tqdm': True,
                          
                          'output_dir': config.output_dir,
                          'logging_dir': config.logging_dir,
                          
                          'fp16': config.fp_16,
                          'fp16_opt_level': config.fp16_opt_level,
                          
                          'optim': config.optim,
                          
                          'num_train_epochs': config.n_epochs,
                          'learning_rate': config.learning_rate,
                          
                          'per_device_train_batch_size': config.batch_size,
                          'per_device_train_batch_size': config.batch_size,

                          'gradient_checkpointing': config.gradient_checkpointing,
                          'gradient_accumulation_steps': config.gradient_accumulation_steps}

    return TrainingArguments(**training_args_dict)



def main(args):
    set_seed(42)
    config = Config(args)
    model = load_model(config)
    tokenizer = T5TokenizerFast.from_pretrained(config.mname, model_max_length=512)

    if config.gradient_checkpointing:
        model.config.use_cache = False

    if args.mode == 'train':
        training_args = set_training_args(config)

        train_dataset = Dataset.from_json('data/train.json')
        valid_dataset = Dataset.from_json('data/valid.json')
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                               model=model, 
                                               padding=True)

        trainer = Trainer(model=model, 
                          args=training_args, 
                          train_dataset=train_dataset, 
                          eval_dataset=valid_dataset,
                          data_collator=data_collator)
        trainer.train()
        model.save(config.ckpt)

    elif args.mode == 'test':
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-strategy', default=0)

    args = parser.parse_args()
    assert args.mode in ['train', 'test']
    assert args.strategy in [i for i in range(5)]

    main(args)