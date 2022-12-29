## Memory_Ablation

Recently, large-scale models are playing a prominent role in various AI tasks. However, as large models have high expressive power, they are expensive to compute. This causes problems both in train process and inference process.

The key to mend this problem is managing memory in efficient fashion. 
This repo presents a set of experiments for efficient GPU memory management.


<br>
<br>

## Training Strategies

**Gradient Accumulate**
> The idea behind gradient accumulation is to instead of calculating the gradients for the whole batch at once to do it in smaller steps. 
The way we do that is to calculate the gradients iteratively in smaller batches by doing a forward and backward pass through the model and accumulating the gradients in the process. 
When enough gradients are accumulated we run the model’s optimization step. 
This way we can easily increase the overall batch size to numbers that would never fit into the GPU’s memory. 
In turn, however, the added forward and backward passes can slow down the training a bit. 
[Click here to check the reference page](https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-accumulation)

<br>

**Gradient Checkpointing**

> Even when we set the batch size to 1 and use gradient accumulation we can still run out of memory when working with large models. 
In order to compute the gradients during the backward pass all activations from the forward pass are normally saved. 
This can create a big memory overhead. Alternatively, one could forget all activations during the forward pass and recompute them on demand during the backward pass. This would however add a significant computational overhead and slow down training.
[Click here to check the reference page](https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing)
<br>

**Mixed precision training**

<br>

**Pruning**

<br>

**Tensor RT**

<br>

**Optimizer**
> The most common optimizer used to train transformer model is Adam or AdamW (Adam with weight decay). 
Adam achieves good convergence by storing the rolling average of the previous gradients which, 
however, adds an additional memory footprint of the order of the number of model parameters. <br>
One remedy to this is to use an alternative optimizer such as **Adafactor**, which works well for some models but often it has instability issues.
Instead of keeping the rolling average for each element in the weight matrices Adafactor only stores aggregated information 
(row- and column-wise sums of the rolling averages) which reduces the footprint considerably. 
One downside of Adafactor is that in some instances convergence can be slower than Adam’s. <br>
Instead of aggregating optimizer states like Adafactor, **8-bit Adam** keeps the full state and quantizes it. 
Quantization means that it stores the state with lower precision and dequantizes it only for the optimization. 
This is similar to the idea behind FP16 training where using variables with lower precision saves memory.
[Click here to check the reference page](https://huggingface.co/docs/transformers/perf_train_gpu_one#optimizer)
<br>
<br>

## Result

<br>
<br>

## Reference

**[LightSeq: A High Performance Inference Library for Transformers](https://arxiv.org/pdf/2010.13887.pdf)**

**[HuggingFace Efficient Training Docs](https://huggingface.co/docs/transformers/perf_train_gpu_one)**
