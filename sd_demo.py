import argparse
import os
import logging
import math
import time
import torch
import lpips

from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from sada import patch
import torchvision.transforms as T

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lora_weights(pipeline, checkpoint_path):
    """
    helper function by adhikjoshi https://github.com/huggingface/diffusers/issues/3064
    """
    from safetensors.torch import load_file
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 0.75
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-2-1')
    parser.add_argument("--prompt", type=str, default="A professional photograph of a red panda.")
    parser.add_argument("--solver", type=str, choices=["euler", "dpm"], default="dpm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args()

    seed = args.seed
    prompt = args.prompt

    lora_path = ""

    baseline_pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")
    if args.solver == "dpm":
        baseline_pipe.scheduler = DPMSolverMultistepScheduler.from_config(baseline_pipe.scheduler.config)
    if args.solver == "euler":
        baseline_pipe.scheduler = EulerDiscreteScheduler.from_config(baseline_pipe.scheduler.config)
    # baseline_pipe = load_lora_weights(baseline_pipe, lora_path)

    logging.info("Warming up GPU...")
    for _ in range(2):
        set_random_seed(seed)
        _ = baseline_pipe(prompt, output_type='pt', height=args.height, width=args.width, num_inference_steps=50).images

    # Baseline
    logging.info("Running baseline...")

    start_time = time.time()
    set_random_seed(seed)
    ori_output = baseline_pipe(prompt, output_type='pt', height=args.height, width=args.width, num_inference_steps=50).images
    use_time = time.time() - start_time

    logging.info("Baseline: {:.2f} seconds".format(use_time))
    del baseline_pipe
    torch.cuda.empty_cache()

    # Cache-Assisted Pruning
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.solver == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.solver == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    # pipe = load_lora_weights(pipe, lora_path)

    patch.apply_patch(pipe,
                    sx=3, sy=3,
                    max_downsample=1,
                    acc_range=(10, 47),

                    lagrange_int=4,
                    lagrange_step=24,
                    lagrange_term=4,

                    max_fix=1024 * 5,
                    max_interval=4
                    )

    logging.info("Warming up GPU...")
    for _ in range(2):
        set_random_seed(seed)
        _ = pipe(prompt, output_type='pt', height=args.height, width=args.width, num_inference_steps=50).images
        patch.reset_cache(pipe)

    logging.info("Running Cache-Assited Pruning...")
    set_random_seed(seed)

    start_time = time.time()
    cap_output = pipe(prompt, output_type='pt', height=args.height, width=args.width, num_inference_steps=50).images
    use_time = time.time() - start_time

    logging.info("CAP: {:.2f} seconds".format(use_time))

    save_image([ori_output[0], cap_output[0]], "output.png")
    logging.info("Saved to output.png. Done!")

    print("Evaluating LPIPS")
    p_r = torch.stack([T.Compose([
        T.Normalize((0.5,), (0.5,))
    ])(img) for img in ori_output]).to('cuda')

    p_o = torch.stack([T.Compose([
        T.Normalize((0.5,), (0.5,))
    ])(img) for img in cap_output]).to('cuda')

    loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')
    d = loss_fn_alex(p_r, p_o)
    print(f"LPIPS: {d.item()}")