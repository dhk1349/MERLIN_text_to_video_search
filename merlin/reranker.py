import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import Optional
import vertexai
from vertexai.vision_models import (
    Image,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
    Video,
    VideoSegmentConfig,
)
from glob import glob
import os


class Reranker(torch.nn.Module):
    def __init__(self, location: str, project_id: str, memory_path: str, queries: list, video_ext: str = ".mp4"):
        super(Reranker, self).__init__()
        self.location = location
        self.project_id = project_id
        self.memory_path = memory_path
        self.video_id = -1
        self.embedding_container = []
        self.weights = [1.5]  # Initialize weights here
        self.video_ext = video_ext  # Store the video extension

        # Load embeddings from JSON file
        # with open(self.memory_path, 'r') as f:
        #     self.memories = json.load(f)
        self.memories = queries

    def get_image_video_text_embeddings(
        self,
        image_path: Optional[str] = None, 
        video_path: Optional[str] = None,
        contextual_text: Optional[str] = None,
        dimension: Optional[int] = 1408,
        video_segment_config: Optional[VideoSegmentConfig] = None,
    ) -> MultiModalEmbeddingResponse:
        """Example of how to generate multimodal embeddings from image, video, and text.

        Args:
            project_id: Google Cloud Project ID, used to initialize vertexai
            location: Google Cloud Region, used to initialize vertexai
            image_path: Path to image (local or Google Cloud Storage) to generate embeddings for.
            video_path: Path to video (local or Google Cloud Storage) to generate embeddings for.
            contextual_text: Text to generate embeddings for.
            dimension: Dimension for the returned embeddings.
                https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings#low-dimension
            video_segment_config: Define specific segments to generate embeddings for.
                https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings#video-best-practices
        Returns:
            MultiModalEmbeddingResponse: A container object holding the embeddings for the provided image, video, and text inputs.
                The embeddings are dense vectors representing the semantic meaning of the inputs.
                Embeddings can be accessed as follows:
                - embeddings.image_embedding (numpy.ndarray): Embedding for the provided image.
                - embeddings.video_embeddings (List[VideoEmbedding]): List of embeddings for video segments.
                - embeddings.text_embedding (numpy.ndarray): Embedding for the provided text.
        """

        vertexai.init(project=self.project_id, location=self.location)

        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
        image, video = None, None
        if image_path is not None:
            image = Image.load_from_file(image_path)
        if video_path is not None:
            video = Video.load_from_file(video_path)

        embeddings = model.get_embeddings(
            image=image,
            video=video,
            video_segment_config=video_segment_config,
            contextual_text=contextual_text,
            dimension=dimension,
        )

        return embeddings

    def init_embedding(self, id):
        # self.video_id = int(id)
        # for memory in self.memories:
        #     if int(memory['id'])==int(id):
        #         self.embedding_container = [memory['vertex_caption']]
        emb_path = glob(os.path.join(self.memory_path, "text_embeddings", f"{id}*.npy"))[0]
        self.embedding_container = [np.load(emb_path)]
        self.weights = [1.5]
        # text_embedding = np.array([memory['vertex_caption'] for memory in self.memories])

        # self.embedding_container = [emb]
        return

    def add_embedding(self, emb, perplexity = None):
        self.embedding_container.append(emb)
        self.weights.append(perplexity)

    def get_embedding(self):
        # return np.mean(self.embedding_container, axis=0)
        # if len(self.embedding_container)==1:
        #     return self.embedding_container[0]
        # weights = [0.8]
        # for i in range(len(self.embedding_container)-1):
        #     weights.append(0.1)
        # return np.average(self.embedding_container, axis=0, weights=weights)
        
        # ppl
        # if len(self.weights) == 0:
        #     return np.average(self.embedding_container, axis=0)
        
        # weights = torch.softmax(torch.tensor(self.weights), dim=-1)
        # weights = 1 / torch.tensor(self.weights)
        # weights = weights.tolist()
        # print(f"weights: {weights}")
        # return np.average(self.embedding_container, axis=0, weights=weights)

        # SLERP
        if len(self.embedding_container) == 1:
            return np.average(self.embedding_container, axis=0)
        alpha = 0.8
        interpolated_vector = self.embedding_container[0]
        for i in self.embedding_container[1:]:
            interpolated_vector = self.slerp(interpolated_vector, i, alpha)
        return interpolated_vector
    
    def get_embedding_slerp(self):
        if len(self.embedding_container) == 1:
            return np.average(self.embedding_container, axis=0)
        alpha = 0.8
        interpolated_vector = torch.tensor(self.embedding_container[0])
        
        # weights = torch.softmax(torch.tensor(self.weights), dim=-1)
        # weights = 1 / torch.tensor(self.weights)
        
        # print(f"weights: {weights}")
        
        for i in self.embedding_container[1:]:
            # print(torch.shape, type(i))
            interpolated_vector = self.slerp(interpolated_vector, i, alpha)
            # interpolated_vector = self.slerp(torch.tensor(i), interpolated_vector, alpha)
            # interpolated_vector = spherical_linear_interpolation(interpolated_vector, torch.tensor(i), alpha) => 이건 성능 낮은 듯 ... slerp_ver_2
        return interpolated_vector
        
    def slerp(self, embedding_A, embedding_B, alpha):
        embedding_A = np.array(embedding_A)
        embedding_B = np.array(embedding_B)

        # Normalize the input vectors
        norm_A = np.linalg.norm(embedding_A)
        norm_B = np.linalg.norm(embedding_B)
        unit_A = embedding_A / norm_A
        unit_B = embedding_B / norm_B

        # Compute the cosine of the angle between the vectors
        dot_product = np.dot(unit_A, unit_B)
        # Numerical stability: ensure the dot product is within the interval [-1.0, 1.0]
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Compute the angle between the vectors
        theta = np.arccos(dot_product)

        # Compute the sin(theta) for the formula
        sin_theta = np.sin(theta)
        
        if sin_theta == 0:
            # If the angle is 0, the two vectors are the same
            return embedding_A

        # Compute the weights for each vector
        weight_A = np.sin((1 - alpha) * theta) / sin_theta
        weight_B = np.sin(alpha * theta) / sin_theta

        # Compute the interpolated vector
        interpolated = weight_A * embedding_A + weight_B * embedding_B

        return interpolated.tolist()

    def rerank(self, target_vid, video_embeddings):
        # target_vid is ground truth. use to find out rank
        # video_embeddings = np.array([memory['vertex'] for memory in self.memories])
        text_embedding = self.get_embedding_slerp()
        # retrieve with new embedding
        similarities = cosine_similarity([text_embedding], video_embeddings)

        # Evaluate top-k retrieval accuracy
        k_values = [10]
        top_k_ids = []
        for k in k_values:
            top_k_indices = np.argsort(-similarities[0])
            for idx, k_index in enumerate(top_k_indices):
                if self.memories[k_index]["video"].replace(self.video_ext, "")==target_vid:
                    desired_video_rank = idx+1
                    break

            top_k_indices = top_k_indices[:k]
            for idx in top_k_indices:
                top_k_ids.append(self.memories[idx]["video"].replace(self.video_ext, ""))
                # top_k_ids.append(self.memories[idx]['id'])
        # desired_video_rank = np.where(np.argsort(-similarities[0]) == [m['id'] for m in self.memories].index(int(target_vid)))[0][0] + 1      
        # desired_video_rank = np.where(np.argsort(-similarities[0]) == [m['video'].replace(".avi", "") for m in self.memories].index(int(target_vid)))[0][0] + 1
        # desired_video_rank = np.where(np.argsort(-similarities[0]) == self.memories.index(self.video_id))[0][0] + 1
        return top_k_ids, desired_video_rank
