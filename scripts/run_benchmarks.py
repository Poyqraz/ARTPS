import argparse
from pathlib import Path
import json
import time


def main():
    parser = argparse.ArgumentParser(description="ARTPS benchmark iskeleti")
    parser.add_argument("--dataset", type=str, required=True, help="Veri seti kökü")
    parser.add_argument("--out", type=str, required=True, help="Sonuç klasörü")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Not: Bu iskelet; gerçek koşturmalar ve metrik hesapları src/eval/* ile tamamlanacak.
    run_info = {
        "dataset": args.dataset,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "placeholder",
    }
    with open(out / "benchmark_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    print("Benchmark iskeleti hazır. Ayrıntılı metrik entegrasyonu eklenecek.")


if __name__ == "__main__":
    main()


