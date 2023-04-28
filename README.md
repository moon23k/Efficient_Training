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

### Common Setups
* **Model** : T5 Small <br>
* **Dataset** : WMT14 En-De Translation Dataset (Train: 10,000 / Valid: 1,000 / Test 1,000 each) <br>
* **Batch Size**: 32 <br>
* **Device**: GPU <br><br>


### Strategy Setups

|  | Strategy_01 | Strategy_02 | Strategy_03 | Strategy_04 | Strategy_05 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Mixed precision training** | N | Y | Y | Y | Y |
| **Gradient accumulation**    | N | N | Y | Y | Y |
| **Gradient checkpointing**   | N | N | N | Y | Y |
| **Optimizer choice**         | N | N | N | N | Y |

<br><br>

## Result

|  | Strategy_01 | Strategy_02 | Strategy_03 | Strategy_04 | Strategy_05 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Time**            | - | - | - | - | - |
| **Best Valid Loss**          | - | - | - | - | - |


<br>
<br>

## Reference
**[LightSeq: A High Performance Inference Library for Transformers](https://arxiv.org/pdf/2010.13887.pdf)** <br>
**[HuggingFace Efficient Training Docs](https://huggingface.co/docs/transformers/perf_train_gpu_one)**
