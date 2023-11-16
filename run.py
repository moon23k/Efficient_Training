import json, argparse, torch
from utils import set_trainer, load_dataset
from transformers import (
    set_seed, AutoTokenizer, 
    AutoModelForSequenceClassification, 
)




class Config(object):
    def __init__(self, strategy):

        self.strategy = strategy        
        self.mname = 'bert-base-uncased'
        
        self.lr = 1e-5
        self.n_epochs = 5
        self.batch_size = 32
        self.max_len = 512
        self.num_labels = 4

        self.ckpt = f"ckpt/{strategy}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def main(strategy):

    #Prerequisites
    set_seed(42)
    config = Config(strategy)

    tokenizer = AutoTokenizer.from_pretrained(
        config.mname, model_max_length=config.max_len
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.mname, num_labels=config.num_labels
    ).to(config.device)

    if config.strategy in ['grad_checkpointing', 'all']:
        model.config.use_cache = False


    #Load datasets
    train_dataset = load_dataset(tokenizer, 'train')
    valid_dataset = load_dataset(tokenizer, 'valid')
    test_dataset = load_dataset(tokenizer, 'test')

    #Load Trainer
    trainer = set_trainer(config, model, tokenizer, train_dataset, valid_dataset)    
    
    #Training
    torch.cuda.reset_max_memory_allocated()
    train_output = trainer.train()
    gpu_memory = torch.cuda.max_memory_allocated()
    
    #Evaluating
    eval_output = trainer.evaluate(test_dataset)
    
    #Save Training and Evaluation Rst Report
    report = {**train_output.metrics, **eval_output}
    report['gpu_memory'] = f"{gpu_memory / (1024 ** 3):.2f} GB"
    with open(f"report/{strategy}.json", 'w') as f:
        json.dump(report, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-strategy', required=True)

    args = parser.parse_args()
    assert args.strategy.lower() in ['vanilla', 'fp16', 'grad_accumulation', 
                                     'grad_checkpointing', 'adafactor', 'all']

    main(args.strategy)