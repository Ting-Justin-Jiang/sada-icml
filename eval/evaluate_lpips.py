import os
import glob
import torch
import lpips
from PIL import Image
from torchvision import transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(device)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

def load_images(folder):
    image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    return image_paths


def get_image_id(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def compute_folder_lpips(original_folder, experiment_folder, batch_size=8):
    ori_paths = load_images(original_folder)
    exp_paths = load_images(experiment_folder)

    ori_dict = {get_image_id(p): p for p in ori_paths}
    exp_dict = {get_image_id(p): p for p in exp_paths}

    matched_ids = sorted(set(ori_dict.keys()).intersection(exp_dict.keys()))
    if len(matched_ids) == 0:
        print("No matching JPG filenames found between the two folders.")
        return None

    lpips_values = []

    for i in range(0, len(matched_ids), batch_size):
        batch_ids = matched_ids[i: i + batch_size]

        ori_tensors = []
        exp_tensors = []

        for file_id in batch_ids:
            ori_img = Image.open(ori_dict[file_id]).convert("RGB")
            exp_img = Image.open(exp_dict[file_id]).convert("RGB")

            ori_tensors.append(transform(ori_img))
            exp_tensors.append(transform(exp_img))

        p_r = torch.stack(ori_tensors).to(device, dtype=torch.float16, non_blocking=True)
        p_o = torch.stack(exp_tensors).to(device, dtype=torch.float16, non_blocking=True)

        with torch.no_grad():
            batch_lpips = lpips_fn(p_r, p_o).cpu()  # Move back to CPU to free GPU memory
            lpips_values.append(batch_lpips)

        del p_r, p_o
        torch.cuda.empty_cache()

    avg_lpips = torch.cat(lpips_values).mean().item()
    return avg_lpips


if __name__ == "__main__":
    original_folder = r"PATH"
    experiment_folder = r"PATH"

    print(original_folder)
    print(experiment_folder)

    avg_lpips = compute_folder_lpips(original_folder, experiment_folder, batch_size=128)

    if avg_lpips is not None:
        print(f"Average LPIPS: {avg_lpips:.4f}")

