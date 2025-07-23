import os
import glob
import torch
from PIL import Image
from torchvision import transforms as T
import clip
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

transform = clip_preprocess

def load_images(folder):
    image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    return image_paths

def get_image_id(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def load_coco_captions():
    dataset = load_dataset("phiyodr/coco2017", split="validation")
    captions = {str(i): sample['captions'][0] for i, sample in enumerate(dataset)}
    return captions


def compute_folder_clipscore(image_folder, coco_captions, batch_size=8):
    image_paths = load_images(image_folder)
    image_ids = [get_image_id(p) for p in image_paths]

    clip_scores = []
    prompts = [coco_captions[img_id] for img_id in image_ids]
    text_tokens = clip.tokenize(prompts).to(device)
    print(prompts)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        batch_text = text_tokens[i: i + batch_size]

        images = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))

        image_input = torch.stack(images).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(batch_text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features * text_features).sum(dim=-1)
            clip_scores.append(similarity.cpu())

        del image_input, batch_text
        torch.cuda.empty_cache()

    avg_clip_score = torch.cat(clip_scores).mean().item()
    return avg_clip_score


if __name__ == "__main__":
    image_folder = r"PATH"
    print(image_folder)

    coco_captions = load_coco_captions()
    avg_clip = compute_folder_clipscore(image_folder, coco_captions, batch_size=64)

    if avg_clip is not None:
        print(f"Average CLIP Score: {avg_clip:.4f}")

