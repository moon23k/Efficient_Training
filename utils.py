import torch, datasets, evaluate
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding    
)




###Data
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.encoding.items()}
        data['labels'] = torch.tensor(self.labels[idx]).long()

        return data

    def __len__(self):
        return len(self.labels)



def load_dataset(tokenizer, split):
    dataset = datasets.Dataset.from_json(f'data/{split}.json')    
    encodings = tokenizer(dataset['x'], padding=True, truncation=True, return_tensors='pt')    
    dataset = Dataset(encodings, dataset['y'])
    return dataset




###Training
def set_trainer(config, model, tokenizer, train_dataset, valid_dataset):

    training_args = TrainingArguments(
        
        output_dir= config.ckpt,
        num_train_epochs= config.n_epochs,
        learning_rate= config.lr,
        per_device_train_batch_size= config.batch_size,
        per_device_eval_batch_size= config.batch_size,
        lr_scheduler_type='reduce_lr_on_plateau',
        load_best_model_at_end= True,

        save_strategy= 'epoch',
        logging_strategy= 'epoch',
        evaluation_strategy= 'epoch',

        fp16= True if config.strategy in ['fp16', 'all'] else False,
        fp16_opt_level= '02' if config.strategy in ['fp16', 'all'] else '01',
        gradient_accumulation_steps = 1 if config.strategy in ['grad_accumulation', 'all'] else 4,
        gradient_checkpointing= True if config.strategy in ['grad_checkpointing', 'all'] else False,
        optim = 'adafactor' if config.strategy in ['optim', 'all'] else 'adamw_torch'
    )


    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return accuracy.compute(predictions=predictions, references=labels)


    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )


    return trainer
