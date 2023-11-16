## Efficient Training

Recently, large-scale models are playing a prominent role in various AI tasks. However, as large models have high expressive power, they are expensive to compute. This causes problems both in train process and inference process.
The key to mend this problem is managing memory in efficient fashion. 
This repo presents a set of experiments for efficient GPU memory management.
In the Experiment, we used pretrained T5-Small Model, and corresponding T5 Tokenizer, and WMT14 Dataset.
For a fair comparison, the rest of the variables except for the training method are fixed equally.

<br><br>

## Training Strategies

**Gradient Accumulating**
> The idea behind gradient accumulation is to instead of calculating the gradients for the whole batch at once to do it in smaller steps. 
The way we do that is to calculate the gradients iteratively in smaller batches by doing a forward and backward pass through the model and accumulating the gradients in the process. When enough gradients are accumulated we run the model’s optimization step. 
This way we can easily increase the overall batch size to numbers that would never fit into the GPU’s memory. 
In turn, however, the added forward and backward passes can slow down the training a bit.

<br>

**Gradient Checkpointing**
> Even when we set the batch size to 1 and use gradient accumulation we can still run out of memory when working with large models. 
In order to compute the gradients during the backward pass all activations from the forward pass are normally saved. 
This can create a big memory overhead. Alternatively, one could forget all activations during the forward pass and recompute them on demand during the backward pass. This would however add a significant computational overhead and slow down training.

<br>

**Mixed precision training**
> The idea of mixed precision training is that not all variables need to be stored in full (32-bit) floating point precision. 
If we can reduce the precision the variales and their computations are faster. 
The main advantage comes from saving the activations in half (16-bit) precision. 
Although the gradients are also computed in half precision they are converted back to full precision for the optimization step so no memory is saved here. Since the model is present on the GPU in both 16-bit and 32-bit precision this can use more GPU memory (1.5x the original model is on the GPU), especially for small batch sizes. Since some computations are performed in full and some in half precision this approach is also called mixed precision training.

<br>

**Adafactor Optimizer**
> The most common optimizer used to train transformer model is Adam or AdamW (Adam with weight decay). Adam achieves good convergence by storing the rolling average of the previous gradients which, however, adds an additional memory footprint of the order of the number of model parameters.
Adafactor is one of the optimizers which can alleviate the aforementioned memory problem. 

<br><br>

## Experimental Setups

### Data and Model Setup
* **Data** <br> 
  For this experiment, small fetched AG_News Dataset has used.  
  Train, Valid, Test Datasets have 1000, 100, 100 volumns each. 
  And all data element has distributed in equal ratio according to labels.
  
* **Model** <br> 
  For this experiment, we utilized the widely recognized NLP model, BERT, with the specific model name being 'bert-base-uncased.' 
  The model comprises a total of 109,485,316 parameters. 
  Apart from the addition of a Linear layer for the classification task, the configuration remains consistent with the default settings of BERT.

<br>

### Training Setup
```
TrainingArguments(
        output_dir= f'ckpt/{strategy}',
        num_train_epochs= 5,
        learning_rate= 1e-5,
        per_device_train_batch_size= 32,
        per_device_eval_batch_size= 32,
        lr_scheduler_type='reduce_lr_on_plateau',
        load_best_model_at_end= True,

        save_strategy= 'epoch',
        logging_strategy= 'epoch',
        evaluation_strategy= 'epoch',

        fp16= True if config.strategy in ['fp16', 'all'] else False,
        fp16_opt_level= '02' if config.strategy in ['fp16', 'all'] else '01',
        gradient_accumulation_steps = True if config.strategy in ['grad_accumulation', 'all'] else 4,
        gradient_checkpointing= True if config.strategy in ['grad_checkpointing', 'all'] else False,
        optim = 'adafactor' if config.strategy in ['optim', 'all'] else 'adamw_torch'
    )
```


<br><br>

## Result

| Training Strategy | Training Time | GPU Occupation | Accuracy |
| :---: | :---: | :---: | :---: |
| Vanilla                | &nbsp; 174 sec &nbsp; (100%) | &nbsp; 7.00 GB &nbsp; (100%) | 79% | 
| FP 16                  | &nbsp;  69 sec &nbsp;  (40%) | &nbsp; 5.47 GB &nbsp;  (78%) | 78% |
| Gradient Accumulation  | &nbsp; 182 sec &nbsp; (105%) | &nbsp; 6.29 GB &nbsp;  (90%) | 83% |
| Gradient Checkpoining  | &nbsp; 239 sec &nbsp; (137%) | &nbsp; 3.54 GB &nbsp;  (51%) | 79% |
| AdaFactor Optimization | &nbsp; 179 sec &nbsp; (103%) | &nbsp; 6.72 GB &nbsp;  (96%) | 79% |
| All Applied            | &nbsp;  85 sec &nbsp;  (49%) | &nbsp; 2.91 GB &nbsp;  (41%) | 80% |


<br><br>

## Reference
**[HuggingFace Efficient Training Docs](https://huggingface.co/docs/transformers/perf_train_gpu_one)**

<br>
