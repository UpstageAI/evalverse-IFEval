import json
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import os
import numpy as np
import ray
from transformers import set_seed, AutoTokenizer
import nltk
nltk.download('punkt')

# Default (ChatML format)

SYSTEM_PREFIX="<|im_start|>system\n"
SYSTEM_SUFFIX="<|im_end|>\n"
USER_PREFIX="<|im_start|>user\n"
USER_SUFFIX="<|im_end|>\n"
ASSISTANT_PREFIX="<|im_start|>assistant\n"
ASSISTANT_SUFFIX="<|im_end|>\n"


def generate_prompts(examples, tokenizer=None):
    if tokenizer is None or tokenizer.chat_template is None:
        print(f"No tokenizer or chat template found, using manual chat template!")
        request = """{SYSTEM_PREFIX}{system_message}{SYSTEM_SUFFIX}{USER_PREFIX}{user_message}{USER_SUFFIX}{ASSISTANT_PREFIX}"""
        prompts = [request.format(SYSTEM_PREFIX=SYSTEM_PREFIX, system_message="", SYSTEM_SUFFIX=SYSTEM_SUFFIX,USER_PREFIX=USER_PREFIX,user_message=example, USER_SUFFIX=USER_SUFFIX,ASSISTANT_PREFIX=ASSISTANT_PREFIX) for example in examples]
    else:
        print(f"Tokenizer with chat template found, using tokenizer.apply_chat_template!")
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": example}], tokenize=False, add_generation_prompt=True) for example in examples]
    return prompts


@ray.remote
def _get_generation_results(data, args, device):
    set_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    model = LLM(model=args.model, tensor_parallel_size=args.gpu_per_inst_eval, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = generate_prompts(data, tokenizer=tokenizer)

    output_list = []

    responses = model.generate(prompts, sampling_params=sampling_params)

    for i, response in enumerate(responses):
        output_list.append({'prompt': data[i], "response": response.outputs[0].text.split("<|im_end|>")[0]})
    return output_list

def run_inst_eval(args):
    set_seed(42)
    if not os.path.exists(f"{args.output_path}/{args.model_name}/ifeval/output.jsonl"):
        ray.init()
        devices = args.devices.split(",")
        num_processes = len(devices) // args.gpu_per_inst_eval
        with open(args.input_data_path) as f:
            raw = f.readlines()
            json_list = [json.loads(j) for j in raw]
        data  = [_json['prompt'] for _json in json_list]
        chunks = np.array_split(data, num_processes)
        jobs = [_get_generation_results.remote(chunk, args, devices[args.gpu_per_inst_eval * i:args.gpu_per_inst_eval *(i + 1)]) for i, chunk in enumerate(chunks)]
        output_lists = ray.get(jobs)

        output = []
        for output_list in output_lists:
            output.extend(output_list)

        os.makedirs(f"{args.output_path}/{args.model_name}/ifeval", exist_ok=True)
        with open(f"{args.output_path}/{args.model_name}/ifeval/output.jsonl" , encoding="utf-8",mode="w") as file:
            for i in output: file.write(json.dumps(i, ensure_ascii=False) + "\n")

    eval_command = f"""python3 evaluation_main.py \
  --input_data={args.input_data_path} \
  --input_response_data={args.output_path}/{args.model_name}/ifeval/output.jsonl \
  --output_dir={args.output_path}/{args.model_name}/ifeval > {args.output_path}/{args.model_name}/ifeval/scores.txt"""
    os.system(eval_command)
def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/project/private/dahyun/checkpoints/SOLAR-1-13B-dev0-instruct")
    parser.add_argument("--model_name", type=str, default="SOLAR-1-13B-dev0-instruct")
    parser.add_argument("--input_data_path", type=str, default="./data/input_data.jsonl")
    parser.add_argument("--output_path", type=str, default="./data") # 끝에 / 없어야 함
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--gpu_per_inst_eval", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    run_inst_eval(args)


if __name__ == "__main__":
    main()
