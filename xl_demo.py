import argparse
import logging
import math
import time
import torch
import lpips

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from torchvision.utils import save_image

from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
import torchvision.transforms as T

from sada import patch


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, default="Ultra‑detailed National Geographic cover photo of a blue macaw perched on a moss‑covered branch, emerald rainforest canopy shimmering with dappled golden sunlight, soft bokeh background, clusters of lavender orchids in the foreground, vibrant saturated colors, shallow depth‑of‑field, shot on an 85mm lens at f/2.8, 8K resolution.")
    parser.add_argument("--solver", type=str, choices=["euler", "dpm"], default="dpm")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lora_path = ""
    seed = args.seed
    prompt = args.prompt

    baseline_pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda:0")
    if args.solver == "dpm":
        baseline_pipe.scheduler = DPMSolverMultistepScheduler.from_config(baseline_pipe.scheduler.config)
    if args.solver == "euler":
        baseline_pipe.scheduler = EulerDiscreteScheduler.from_config(baseline_pipe.scheduler.config)
    # baseline_pipe.load_lora_weights(lora_path)

    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(1):
        set_random_seed(seed)
        _ = baseline_pipe(prompt, num_inference_steps=50, output_type='pt').images

    # Baseline
    logging.info("Running baseline...")
    start_time = time.time()
    set_random_seed(seed)

    ori_output = baseline_pipe(prompt, num_inference_steps=50, output_type='pt').images
    baseline_use_time = time.time() - start_time
    logging.info("Baseline: {:.2f} seconds".format(baseline_use_time))

    del baseline_pipe
    torch.cuda.empty_cache()

    # CAP
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda:0")
    if args.solver == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.solver == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    # pipe.load_lora_weights(lora_path)


    patch.apply_patch(pipe,
                      sx=3, sy=3,
                      max_downsample=2,
                      acc_range=(10, 47),

                      lagrange_int=4,
                      lagrange_step=24,
                      lagrange_term=4,

                      max_fix=1024 * 10,
                      max_interval=4)


    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(1):
        set_random_seed(seed)
        _ = pipe(prompt, num_inference_steps=50, output_type='pt').images
        patch.reset_cache(pipe)

    logging.info("Running CAP...")
    set_random_seed(seed)
    start_time = time.time()

    cap_output = pipe(prompt, num_inference_step=50, output_type='pt').images
    use_time = time.time() - start_time
    logging.info("CAP: {:.2f} seconds".format(use_time))

    logging.info("Baseline: {:.2f} seconds. CAP: {:.2f} seconds".format(baseline_use_time, use_time))
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