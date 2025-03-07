import json
import numpy as np
from glob import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass

from typing import Optional

# Add the project root to Python path when running as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

from utils.logger import logger

class DatasetConfig(NamedTuple):
    """Configuration for dataset-specific file paths and settings."""
    name: str
    test_file: str
    video_ext: str
    video_subdirs: List[str]
    additional_files: Dict[str, str]
    sample_limit: Optional[int] = None

# Dataset-specific configurations
DATASET_CONFIGS = {
    "msvd": DatasetConfig(
        name="msvd",
        test_file="msvd_ret_test.json",
        video_ext=".avi",
        video_subdirs=["YouTubeClips"],
        additional_files={},
        sample_limit=None
    ),
    "msrvtt": DatasetConfig(
        name="msrvtt",
        test_file="test_videodatainfo.json",
        video_ext=".mp4",
        video_subdirs=["videos/TestVideo", "videos/TrainValVideo"],
        additional_files={
            "train_val": "train_val_videodatainfo.json",
            "msrvtt_1ka": "msrvtt_1ka"
        },
        sample_limit=500
    ),
    "anet": DatasetConfig(
        name="anet",
        test_file="descs_ret_test.json",
        video_ext=".mp4",
        video_subdirs=["videos"],
        additional_files={},
        sample_limit=None
    )
}

@dataclass
class DatasetPaths:
    """Holds paths for dataset files and directories."""
    base_path: Path
    video_paths: List[Path]  # List of video paths
    gpt4v_caption_path: Path
    video_embeddings_path: Path
    text_embeddings_path: Path
    config: DatasetConfig
    
    @classmethod
    def from_base_path(cls, base_path: Union[str, Path], config: DatasetConfig) -> 'DatasetPaths':
        """Create DatasetPaths from a base directory path."""
        base = Path(base_path)
        # Create a list of video paths from the video_subdirs list
        video_paths = [base / subdir for subdir in config.video_subdirs]
        
        return cls(
            base_path=base,
            video_paths=video_paths,
            gpt4v_caption_path=base / "gpt4v_captions",
            video_embeddings_path=base / "video_embeddings",
            text_embeddings_path=base / "text_embeddings",
            config=config
        )

    def get_video_paths(self) -> List[Path]:
        """Get all video paths."""
        return self.video_paths
        
    def find_video_path(self, video_id: str) -> Optional[Path]:
        """
        Find the path to a video file by checking all video subdirectories.
        
        Args:
            video_id: The ID of the video
            
        Returns:
            Path to the video file if found, None otherwise
        """
        for video_path in self.video_paths:
            full_path = video_path / f"{video_id}{self.config.video_ext}"
            if full_path.exists():
                return full_path
        return None

    def get_test_file_path(self) -> Path:
        """Get the path to the test file."""
        return self.base_path / self.config.test_file

    def get_additional_file_path(self, file_key: str) -> Path:
        """Get the path to an additional dataset file."""
        if file_key not in self.config.additional_files:
            raise ValueError(f"Unknown additional file key: {file_key}")
        return self.base_path / self.config.additional_files[file_key]

def load_video_captions(caption_path: Path) -> Dict[str, Any]:
    """
    Load video captions from json files in the specified directory.
    
    Args:
        caption_path: Path to directory containing caption files
        
    Returns:
        Dictionary mapping video IDs to their captions
    """
    video_captions = {}
    for vc in glob(str(caption_path / "*")):
        with open(vc, 'r') as f:
            caption = json.load(f)
        video_id = Path(vc).stem.replace(".npy", "")  # Handle both .json and .npy files
        video_captions[video_id] = caption
    return video_captions

def load_embeddings(video_id: str, paths: DatasetPaths) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load video and text embeddings for a given video ID.
    
    Args:
        video_id: ID of the video
        paths: DatasetPaths object containing relevant paths
        
    Returns:
        Tuple of (video_embedding, text_embedding)
    """
    # Load video embedding
    video_emb_path = paths.video_embeddings_path / f"{video_id}.npy"
    video_emb = np.load(str(video_emb_path))
    
    # Load text embedding
    text_emb_path = next(paths.text_embeddings_path.glob(f"{video_id}*"))
    text_emb = np.load(str(text_emb_path))
    
    return video_emb, text_emb

def prepare_msvd_data(paths: DatasetPaths, config: DatasetConfig) -> Tuple[List[Dict], Dict, List[np.ndarray], List[np.ndarray]]:
    """Handle MSVD dataset loading."""
    # Load queries
    logger.info("Loading MSVD dataset...")
    with open(paths.get_test_file_path(), 'r') as f:
        queries = json.load(f)

    # Load video captions
    logger.info("Loading video meta data...")
    video_captions = load_video_captions(paths.gpt4v_caption_path)

    # Load embeddings
    logger.info("Loading embeddings...")
    video_embs = []
    text_embs = []
    
    for idx, q in enumerate(queries):
        video_id = q['video'].replace(config.video_ext, "")
        video_emb, text_emb = load_embeddings(video_id, paths)
        video_embs.append(video_emb)
        text_embs.append(text_emb)
        
        # Update query caption
        text_emb_path = next(paths.text_embeddings_path.glob(f"{video_id}*"))
        query_idx = int(text_emb_path.stem.split("_")[-1])
        queries[idx]['caption'] = queries[idx]['caption'][query_idx]
        
    return queries, video_captions, video_embs, text_embs

def prepare_msrvtt_data(paths: DatasetPaths, config: DatasetConfig) -> Tuple[List[Dict], Dict, List[np.ndarray], List[np.ndarray]]:
    """Handle MSRVTT dataset loading."""
    # Load queries
    sampled_queries = {}
    
    # Load train/val data
    with open(paths.get_additional_file_path("train_val"), 'r') as f:
        train_val_files = json.load(f)['sentences']
    for sample in train_val_files:
        sampled_queries[sample["video_id"]] = sample["caption"]

    # Load test data
    with open(paths.get_test_file_path(), 'r') as f:
        test_files = json.load(f)['sentences']
    for sample in test_files:
        sampled_queries[sample["video_id"]] = sample["caption"]

    # Load MSRVTT1KA IDs
    with open(paths.get_additional_file_path("msrvtt_1ka"), 'r') as f:
        msrvtt1ka_ids = [line.strip() for line in f]

    # Create queries list with sample limit
    queries = [
        {"video": id, "caption": sampled_queries[id]} 
        for id in msrvtt1ka_ids[:config.sample_limit]
    ]

    # Load video captions and embeddings
    logger.info("Loading video meta data...")
    video_captions = load_video_captions(paths.gpt4v_caption_path)

    logger.info("Loading embeddings...")
    video_embs = []
    text_embs = []
    
    for q in queries:
        video_id = q['video']
        video_emb, text_emb = load_embeddings(video_id, paths)
        video_embs.append(video_emb)
        text_embs.append(text_emb)

    return queries, video_captions, video_embs, text_embs

def prepare_anet_data(paths: DatasetPaths, config: DatasetConfig) -> Tuple[List[Dict], Dict, List[np.ndarray], List[np.ndarray]]:
    """Handle ActivityNet dataset loading."""
    # Load queries
    with open(paths.get_test_file_path(), 'r') as f:
        test_files = json.load(f)

    sampled_queries = {
        sample["video_id"]: sample["desc"]
        for sample in test_files
        if "video_id" in sample and "desc" in sample
    }
    
    # Create queries list
    queries = [
        {"video": id, "caption": desc} 
        for id, desc in sampled_queries.items()
    ]
    
    # Apply sample limit if specified
    if config.sample_limit:
        queries = queries[:config.sample_limit]

    # Load video captions
    logger.info("Loading video meta data...")
    video_captions = load_video_captions(paths.gpt4v_caption_path)
    
    # Load embeddings
    logger.info("Loading embeddings...")
    video_embs = []
    text_embs = []
    
    for q in queries:
        video_id = q['video']
        video_emb, text_emb = load_embeddings(video_id, paths)
        video_embs.append(video_emb)
        text_embs.append(text_emb)

    return queries, video_captions, video_embs, text_embs

def prepare_data(
    dataset: str,
    video_path: Union[str, Path],
    caption: Any = None
) -> Tuple[List[Dict], Dict, List[np.ndarray], List[np.ndarray]]:
    """
    Prepare data for a specified dataset.
    
    Args:
        dataset: Name of the dataset ('msvd', 'msrvtt', or 'anet')
        video_path: Path to the dataset directory
        caption: Optional caption data (currently unused)
        
    Returns:
        Tuple containing:
        - queries: List of query dictionaries
        - video_captions: Dictionary of video captions
        - video_embs: List of video embeddings
        - text_embs: List of text embeddings
        
    Raises:
        ValueError: If dataset is not supported
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset} is not supported")
    
    config = DATASET_CONFIGS[dataset]
    paths = DatasetPaths.from_base_path(video_path, config)
    
    dataset_handlers = {
        "msvd": prepare_msvd_data,
        "msrvtt": prepare_msrvtt_data,
        "anet": prepare_anet_data
    }
    
    return dataset_handlers[dataset](paths, config)