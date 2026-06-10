import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

try:
    from src.core import CuriosityScorer
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.core import CuriosityScorer


def _safe_float(v, default=0.0):
	try:
		if v is None:
			return float(default)
		return float(v)
	except Exception:
		return float(default)


def main():
	parser = argparse.ArgumentParser(description="Curiosity ağırlıklarını öğren (basit)")
	parser.add_argument("--features", type=str, required=True, help="Özellik dosyası (JSON satırları veya tek JSON listesi)")
	parser.add_argument("--out", type=str, required=True, help="Model çıkışı (weights.json)")
	args = parser.parse_args()

	feat_path = Path(args.features)
	data: List[Dict] = []
	with open(feat_path, 'r', encoding='utf-8') as f:
		txt = f.read().strip()
		try:
			obj = json.loads(txt)
			if isinstance(obj, list):
				data = obj
			else:
				data = [obj]
		except Exception:
			# satır bazlı JSON
			data = [json.loads(line) for line in txt.splitlines() if line.strip()]

	X = []
	y = []
	for row in data:
		# Beklenen alanlar: known_value_score, anomaly_mse, combined_anomaly_score,
		# depth_variance, roughness, label (0-1)
		X.append({
			'known_value_score': _safe_float(row.get('known_value_score', 0.0), 0.0),
			'anomaly_mse': _safe_float(row.get('anomaly_mse', 0.0), 0.0),
			'combined_anomaly_score': _safe_float(row.get('combined_anomaly_score', 0.0), 0.0),
			'depth_variance': _safe_float(row.get('depth_variance', 0.0), 0.0),
			'roughness': _safe_float(row.get('roughness', 0.0), 0.0),
		})
		y.append(_safe_float(row.get('label', 0.0), 0.0))

	scorer = CuriosityScorer()
	weights = scorer.fit(X, y)
	out = Path(args.out)
	out.parent.mkdir(parents=True, exist_ok=True)
	scorer.save(str(out))
	print("Öğrenilen ağırlıklar:", weights)


if __name__ == "__main__":
	main()


