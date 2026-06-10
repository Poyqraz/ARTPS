import argparse
from pathlib import Path
import json
from typing import Optional
from src.data.pds_client import PDSAtlasClient, PDSAtlasQuery


def _is_image_name(name: str) -> bool:
	name = name.lower()
	return any(name.endswith(ext) for ext in [
		".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".img"
	])


def main():
	parser = argparse.ArgumentParser(description="PDS veri setini indir ve hazırla (Atlas API)")
	parser.add_argument("--out", type=str, required=True, help="Çıkış klasörü")
	parser.add_argument("--instrument", type=str, default="Mastcam-Z", help="Enstrüman adı (örn. Mastcam-Z, Mastcam)")
	parser.add_argument("--host", type=str, default="Perseverance", help="Enstrüman host/rover (örn. Perseverance, Curiosity)")
	parser.add_argument("--target", type=str, default="Mars", help="Hedef (örn. Mars)")
	parser.add_argument("--product_type", type=str, default=None, help="Ürün tipi filtresi (örn. calibrated)")
	parser.add_argument("--sol_start", type=int, default=None, help="Başlangıç SOL")
	parser.add_argument("--sol_end", type=int, default=None, help="Bitiş SOL")
	parser.add_argument("--rows", type=int, default=200, help="Sayfa başına kayıt sayısı")
	parser.add_argument("--max_pages", type=int, default=5, help="En fazla sayfa sayısı")
	parser.add_argument("--download", action="store_true", help="Aday görüntüleri indir")
	args = parser.parse_args()

	out = Path(args.out)
	(out / "raw").mkdir(parents=True, exist_ok=True)
	(out / "images").mkdir(parents=True, exist_ok=True)
	(out / "metadata").mkdir(parents=True, exist_ok=True)
	(out / "splits").mkdir(parents=True, exist_ok=True)

	client = PDSAtlasClient()

	planet_range: Optional[tuple[int, int]] = None
	if args.sol_start is not None and args.sol_end is not None:
		planet_range = (int(args.sol_start), int(args.sol_end))

	all_docs = []
	for page in range(max(1, int(args.max_pages))):
		start = page * max(1, int(args.rows))
		q = PDSAtlasQuery(
			instrument=args.instrument,
			instrument_host=args.host,
			target=args.target,
			product_type=args.product_type,
			planet_day_range=planet_range,
			rows=int(args.rows),
			start=start,
		)
		try:
			payload = client.search(q)
			docs = client.extract_docs(payload)
			if not docs:
				break
			all_docs.extend(docs)
			# sayfa dolmadıysa dur
			if len(docs) < int(args.rows):
				break
		except Exception as e:
			print("Atlas sorgusu başarısız:", e)
			break

	# kayıtları kaydet
	with open(out / "metadata" / "atlas_docs.json", "w", encoding="utf-8") as f:
		json.dump(all_docs, f, indent=2, ensure_ascii=False)
	print(f"Toplam {len(all_docs)} kayıt alındı.")

	# indirme opsiyonu
	if args.download and all_docs:
		img_dir = out / "images"
		import requests
		sess = requests.Session()
		dl_count = 0
		for d in all_docs:
			urls = client.candidate_urls(d)
			# basit filtre: dosya adı görüntü uzantısı içeriyor mu?
			img_urls = [u for u in urls if _is_image_name(u)]
			for u in img_urls:
				local = client.try_download({'href': u}, img_dir, session=sess)
				if local is not None:
					dl_count += 1
					break
		print(f"İndirilen dosya: {dl_count}")

	# dataset kartı
	structure = {
		"instrument": args.instrument,
		"host": args.host,
		"target": args.target,
		"product_type": args.product_type,
		"planet_day_range": planet_range,
		"total_docs": len(all_docs),
	}
	with open(out / "DATASET_CARD.json", "w", encoding="utf-8") as f:
		json.dump(structure, f, indent=2, ensure_ascii=False)
	print("Dataset hazır:", out)


if __name__ == "__main__":
	main()


