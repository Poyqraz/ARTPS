import json
from huggingface_hub import scan_cache

if __name__ == "__main__":
	report = scan_cache()
	targets = {
		"damo-vilab/text-to-video-ms-1.7b",
		"stabilityai/stable-video-diffusion-img2vid-xt",
	}
	print("Hugging Face cache taraması:\n")
	for repo in report.repos:
		if repo.repo_id in targets:
			print(f"- Repo: {repo.repo_id}")
			print(f"  Tip: {repo.repo_type}")
			print(f"  Toplam Boyut: {round(repo.size_on_disk / (1024**3), 2)} GB")
			for rev in repo.revisions:
				print(f"    Rev: {rev.commit_hash}  Boyut: {round(rev.size_on_disk / (1024**3), 2)} GB")
			print("")
	print("Bitti.")

