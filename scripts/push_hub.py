import dataclasses
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

@dataclasses.dataclass
class Args:
    repo_id: str  # Replace with your repo_id
    root: str
if __name__ == "__main__":
    args = tyro.cli(Args)

    dataset = LeRobotDataset(args.repo_id, args.root)

    # 上传到 Hugging Face Hub
    dataset.push_to_hub(args.repo_id)