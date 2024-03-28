# IFEval: Instruction Following Eval

This is not an officially supported Google product.

## Dependencies

Please make sure that all required python packages are installed via:

```
pip install -r requirements.txt
```

## How to run

We will use vLLM to generate responses for the instruction prompts via the python file `inst_eval.py`

```bash
python inst_eval.py \
--model {ckpt_path} --model_ref_id {model_ref_id} \
--output_path {ckpt_path}/eval_vllm \
```

- ckpt_path: Path to the model checkpoints, not ending with `/`.
- model_ref_id: A shorthand name for the model. This will be used in the path to save the evaluation results.

At the moment, you can specify `--devices` and `--gpu_per_inst_eval` to set total number of GPUs and GPUs per inst_eval process (e.g. vLLM).
However, as there are slight variations with differing number of GPUs and GPUs per inst_eval process, using the default value of `--devices` and `--gpu_per_inst_eval` is recommended for reproducible evaluation results.
