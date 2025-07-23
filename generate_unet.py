import time
import argparse
import numpy as np
import random

import os
from tqdm import tqdm

import torch
from datasets import load_dataset
from torchvision.transforms.functional import to_pil_image
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler

from sada import patch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    if args.dataset == 'parti':
        prompts = load_dataset("nateraw/parti-prompts", split="train")
    elif args.dataset == 'coco2017':
        dataset = load_dataset("phiyodr/coco2017")
        prompts = [{"Prompt": sample['captions'][0]} for sample in dataset['validation']]
    else:
        raise NotImplementedError

    prompts = prompts[:args.num_fid_samples]

    if args.model == "stabilityai/stable-diffusion-2-1":
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
        if args.solver == "dpm":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if args.solver == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        max_downsample = 1

    elif args.model == "stabilityai/stable-diffusion-xl-base-1.0":
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
        if args.solver == "dpm":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if args.solver == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        max_downsample = 0

    else: raise NotImplementedError

    if args.method == 'sada':
        patch.apply_patch(pipe,
                          ratio=0.99,
                          mode="cache_merge",
                          sx=3, sy=3,
                          max_downsample=max_downsample,
                          acc_range=(args.acc_start, args.acc_end),
                          prune=True,

                          interp_mode="x_0",

                          lagrange_int=args.lagrange_int,
                          lagrange_step=args.lagrange_step,
                          lagrange_term=args.lagrange_term,

                          max_fix=args.max_fix,
                          max_interval=args.max_interval)

    output_dir = args.experiment_folder
    os.makedirs(output_dir, exist_ok=True)

    num_batch = len(prompts) // args.batch_size
    if len(prompts) % args.batch_size != 0:
        num_batch += 1

    global_image_index = 0  # Tracks unique image indices across batches
    use_time = 0

    for i in tqdm(range(num_batch)):
        start, end = args.batch_size * i, min(args.batch_size * (i + 1), len(prompts))
        sample_prompts = [prompts[i]["Prompt"] for i in range(start, end)]

        set_random_seed(args.seed)
        start_time = time.time()
        if args.method != "deep_cache":
            pipe_output = pipe(
                sample_prompts, output_type='np', return_dict=True,
                num_inference_steps=args.steps
            )
        else:
            pipe_output = pipe(
                sample_prompts, num_inference_steps=args.steps,
                cache_interval=args.update_interval,
                cache_layer_id=args.layer, cache_block_id=args.block,
                uniform=args.uniform, pow=args.pow, center=args.center,
                output_type='np', return_dict=True
            )
        use_time += round(time.time() - start_time, 2)
        images = pipe_output.images

        for image in images:
            image = to_pil_image((image * 255).astype(np.uint8))  # Convert to PIL image
            image.save(f"{output_dir}/{global_image_index}.jpg")  # Use global index
            global_image_index += 1

        if args.method == 'sada':
            patch.reset_cache(pipe)

    print(f"Done: use_time = {use_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # == Sampling setup ==
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-xl-base-1.0')
    parser.add_argument("--dataset", type=str, default="coco2017")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-fid-samples", type=int, default=5000)
    parser.add_argument('--experiment-folder', type=str, default='samples/inference/sada')
    parser.add_argument("--solver", type=str, choices=["euler", "dpm"], default="dpm")

    # == Acceleration Setup ==
    parser.add_argument("--method", type=str, choices=["original", "sada"], default="sada")

    parser.add_argument("--max-fix", type=int, default=1024 * 5)
    parser.add_argument("--max-interval", type=int, default=4)

    parser.add_argument("--acc-start", type=int, default=10)
    parser.add_argument("--acc-end", type=int, default=47)

    parser.add_argument("--lagrange-term", type=int, default=4)
    parser.add_argument("--lagrange-step", type=int, default=24)
    parser.add_argument("--lagrange-int", type=int, default=4)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)