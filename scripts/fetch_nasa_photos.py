import argparse
from pathlib import Path
from src.data.nasa_mars_photos import NASAPhotosClient


def main():
    p = argparse.ArgumentParser(description="NASA Mars Photos indirme aracı")
    p.add_argument("--rover", type=str, default="curiosity")
    p.add_argument("--camera", type=str, default="mast")
    p.add_argument("--sol_start", type=int, required=True)
    p.add_argument("--sol_end", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max_per_sol", type=int, default=10)
    args = p.parse_args()

    client = NASAPhotosClient()
    total = client.fetch_range(
        rover=args.rover,
        sol_start=args.sol_start,
        sol_end=args.sol_end,
        out_dir=Path(args.out),
        camera=args.camera,
        max_per_sol=int(args.max_per_sol),
    )
    print(f"Toplam indirilen: {total} dosya")


if __name__ == "__main__":
    main()


