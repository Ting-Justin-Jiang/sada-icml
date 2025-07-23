import argparse
from pytorch_fid import fid_score

def main():
    parser = argparse.ArgumentParser(description='Compute FID between original and cap folders')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for InceptionV3 inference')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0 or cpu)')
    args = parser.parse_args()

    original_folder = r"PATH"
    experiment_folder = r"PATH"

    # Compute FID score
    fid_value = fid_score.calculate_fid_given_paths(
        paths=[original_folder, experiment_folder],
        batch_size=args.batch_size,
        device=args.device,
        dims=2048  # Standard InceptionV3 feature dimension
    )

    print(f'\nFID Score: {fid_value:.2f}')
    print(f'  Original: {original_folder}')
    print(f'  SADA:      {experiment_folder}')

if __name__ == '__main__':
    main()