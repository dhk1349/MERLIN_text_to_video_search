import io
import os
import cv2
import math
import json
import time
import torch
import base64
import openai
import asyncio
import aiohttp
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image as Img
import argparse
from pathlib import Path
import sys
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_utils import prepare_data
from utils.video_utils import video_frame_generator
from utils.logger import logger, setup_logger
from utils.setup_directories import verify_structure, create_directory_structure
from utils.env_utils import load_env_variables, get_required_env, get_optional_env

from typing import Optional

import vertexai
from vertexai.vision_models import (
    Image,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
    Video,
    VideoSegmentConfig,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class Answerer(torch.nn.Module):
    def __init__(self, api_key):
        super(Answerer, self).__init__()
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        # Load VQA model
        # config_path = './mPLUG/configs_video/VideoQA_msrvtt_large.yaml'
        # checkpoint_path = './mPLUG2_MSRVTT_QA.pth'
        # video_qa_model = VideoQAModel(config_path, checkpoint_path)

        # Load VQA model-> BLIP2 + ChatGPT3.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.vqa_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.vqa_model = "gpt-4o"
        # self.vqa_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
        # self.vqa_model.to(self.device)
        self.vqa_sys_prompt = {
        "role": "system",
        "content": """
        You are a helpful assistant that answer the question with details. Don't jsut answer is yes or no. Provide more details(about facts) about the image that might help the questioner.
        """,
        }

        self.vqa_sys_prompt_language_only = {
        "role": "system",
        "content": """
        You are a helpful assistant that answer the question with details. Don't jsut answer is yes or no. Provide more details(about facts) about the image that might help the questioner.
        """,
        }

        self.system_prompt = {
        "role": "system",
        "content": """
        The VQA model is designed to answer questions based on images. 
        To apply it to videos, frames are uniformly extracted from the video over time, and the model provides an answer for each frame to a given question. 
        This means that for a single question, there will be multiple answers - one for each extracted frame. 
        Your role is to review all of the individual answers and summarize them to provide a final answer to the original question. 
        When making final answer, don't user unnecessary words like 'Based on the individual answers provided by the VQA model,'. Just answer to the question.

        For example, if the question is "Did a cookie appear in the video?" and the individual answers from the frames are ["No", "No", "Yes", "No"], 
        then since a cookie appeared in the 3rd frame, you should summarize and answer the question as "Yes".
        Length of aggregated answer should be around 30~35 words.
        """,
        } 
        
        self.model = "gpt-4o" # "gpt-4-1106-preview" "gpt-3.5-turbo-16k"
        self.images = []
        self.topk = []

    def load_topk(self, topk):
        self.topk = topk

    def load_video(self, video_path):
        self.images = [] 
        for frame in video_frame_generator(video_path, resize_factor=1.0, skip_second=1):
            self.images.append(frame)
        self.images = [self.encode_image(img) for img in self.images]

    def encode_image(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode("utf-8")

    def ask(self, question):
        print("Asking..")
        answers = []
        for img in self.images:
            response = self.client.chat.completions.create(
            model=self.vqa_model,
            messages=[
                self.vqa_sys_prompt,
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img}"}
                    }
                ]}
            ],
            max_tokens=50,
            temperature=0.3,
            )
            answers.append(response.choices[0].message.content)

        ans = self.aggregate(question, answers)

        return ans, answers
    
    async def async_ask(self, question):
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
        async def fetch_answer(session, question, img):
            payload = {
                "model": self.vqa_model,
                "messages": [
                    self.vqa_sys_prompt,
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                        ]
                    }
                ],
                "max_tokens": 50,
                "temperature": 0.3,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # Add your API key here
            }
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as response:
                response_json = await response.json()
                try: 
                    return response_json["choices"][0]["message"]["content"]
                except:
                    return " "

        answers = None
        async with aiohttp.ClientSession() as session:
            try:
                tasks = [fetch_answer(session, question, img) for img in self.images]
                answers = await asyncio.gather(*tasks)
            except:
                print("Answer failed..")
                
            if answers!=None:
                ans = self.aggregate(question, answers)
            else: 
                ans = "no answer available"
                answers = ["no answer available", "no answer available", "no answer available"]
        return ans, answers
    
    def aggregate(self, question, answers):
        aggregation_prompt = {"role": "user", "content": f""" 
            Question: {question}\n
            VQA answers: {answers}\n
            Agregated Answer: 
            """}
        messages = [self.system_prompt, aggregation_prompt]
        
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=100, stream=False, temperature=0.5
        )
        response = completion.choices[0].message.content
        return response

class Reranker(torch.nn.Module):
    def __init__(self, location: str, project_id: str, memory_path: str, queries: list):
        super(Reranker, self).__init__()
        self.location = location
        self.project_id = project_id
        self.memory_path = memory_path
        self.video_id = -1
        self.embedding_container = []

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
        emb_path = glob(os.path.join(self.memory_path, "text_embeddings", f"{id}*.npy"))[0]
        self.embedding_container = [np.load(emb_path)]

        # text_embedding = np.array([memory['vertex_caption'] for memory in self.memories])

        # self.embedding_container = [emb]
        return

    def add_embedding(self, emb):
        self.embedding_container.append(emb)

    def get_embedding(self):
        # return np.mean(self.embedding_container, axis=0)
        weights = [0.8]
        for i in range(len(self.embedding_container)-1):
            weights.append(0.1)
        return np.average(self.embedding_container, axis=0, weights=weights)
        
    def rerank(self, target_vid, video_embeddings):
        # target_vid is ground truth. use to find out rank
        # video_embeddings = np.array([memory['vertex'] for memory in self.memories])
        text_embedding = self.get_embedding()
        # retrieve with new embedding
        similarities = cosine_similarity([text_embedding], video_embeddings)

        # Evaluate top-k retrieval accuracy
        k_values = [10]
        top_k_ids = []
        for k in k_values:
            top_k_indices = np.argsort(-similarities[0])
            for idx, k_index in enumerate(top_k_indices):
                if self.memories[k_index]["video"].replace(".avi", "")==target_vid:
                    desired_video_rank = idx+1
                    break

            top_k_indices = top_k_indices[:k]
            for idx in top_k_indices:
                top_k_ids.append(self.memories[idx]["video"].replace(".avi", ""))
                # top_k_ids.append(self.memories[idx]['id'])
        # desired_video_rank = np.where(np.argsort(-similarities[0]) == [m['id'] for m in self.memories].index(int(target_vid)))[0][0] + 1
        
        # desired_video_rank = np.where(np.argsort(-similarities[0]) == [m['video'].replace(".avi", "") for m in self.memories].index(int(target_vid)))[0][0] + 1
        # desired_video_rank = np.where(np.argsort(-similarities[0]) == self.memories.index(self.video_id))[0][0] + 1
        return top_k_ids, desired_video_rank

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MERLIN: Multimodal Embedding Refinement via LLM-based Iterative Navigation"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["msvd", "msrvtt", "anet"],
        required=True,
        help="Dataset to process"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to dataset directory"
    )
    
    # Model configuration (can be overridden from .env)
    parser.add_argument(
        "--model_name",
        type=str,
        help="OpenAI model to use (default from .env)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Maximum tokens for model response (default from .env)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default from .env)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    # Environment configuration
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to .env file"
    )
    
    return parser.parse_args()

def setup_environment(args: argparse.Namespace) -> None:
    """
    Set up the environment for running MERLIN.
    
    Args:
        args: Parsed command line arguments
    """
    # Load environment variables
    load_env_variables(args.env_file)
    
    # Set up logging
    log_level = args.log_level or get_optional_env("LOG_LEVEL", "INFO")
    setup_logger(level=log_level)
    
    # Set debug mode from environment if not set in args
    if not args.debug and get_optional_env("DEBUG", "false").lower() == "true":
        args.debug = True
    
    # Verify/create directory structure
    if not verify_structure():
        logger.info("Creating directory structure...")
        create_directory_structure(debug=args.debug)
    
    # Set up model configuration from environment if not provided in args
    if not args.model_name:
        args.model_name = get_optional_env("MODEL_NAME", "gpt-4-vision-preview")
    if not args.max_tokens:
        args.max_tokens = int(get_optional_env("MAX_TOKENS", "300"))
    
    # Get required API key from environment
    api_key = get_required_env("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Set up Google Cloud configuration
    os.environ["GOOGLE_CLOUD_PROJECT_ID"] = get_required_env("GOOGLE_CLOUD_PROJECT_ID")
    os.environ["GOOGLE_CLOUD_LOCATION"] = get_required_env("GOOGLE_CLOUD_LOCATION")

def main() -> None:
    """Main entry point for MERLIN."""
    args = parse_args()
    
    try:
        # Set up environment
        setup_environment(args)
        
        # Load dataset
        logger.info(f"Processing dataset: {args.dataset}")
        queries, video_captions, video_embs, text_embs = prepare_data(
            dataset=args.dataset,
            video_path=args.data_path,
            caption=None  # TODO: Implement caption handling if needed
        )
        
        # Retrieve top-5 video candidates based on cosine similarity
        logger.info("Zero-shot retrieval evaluation...")
        top_k = 10
        predictions = []

        for i, query_text_emb in tqdm(enumerate(text_embs)):
            similarities = cosine_similarity([query_text_emb], video_embs)
            top_k_indices = np.argsort(-similarities[0])[:top_k]
            
            # Create a dictionary for the prediction
            prediction = {
                "query_id": queries[i]["video"].replace(".avi", ""),
                "org_ranking": [queries[j]["video"].replace(".avi", "") for j in top_k_indices.squeeze()]
            }
            predictions.append(prediction)

        # Calculate top-1, top-5, and top-10 retrieval accuracies
        top_1_acc = 0
        top_5_acc = 0
        top_10_acc = 0
        for pred in predictions:
            target_vid = pred["query_id"]
            ranking = pred["org_ranking"]
            if target_vid == ranking[0]:
                top_1_acc += 1
            if target_vid in ranking[:5]:
                top_5_acc += 1
            if target_vid in ranking[:10]:
                top_10_acc += 1

        total_queries = len(predictions)
        logger.info(f"Top-1 retrieval accuracy: {top_1_acc / total_queries * 100:.2f}% {top_1_acc}/{total_queries}")
        logger.info(f"Top-5 retrieval accuracy: {top_5_acc / total_queries * 100:.2f}% {top_5_acc}/{total_queries}")
        logger.info(f"Top-10 retrieval accuracy: {top_10_acc / total_queries * 100:.2f}% {top_10_acc}/{total_queries}")

        video_base_path = f"{args.data_path}/YouTubeClips"
        video_caption_path = f"{args.data_path}/gpt4o_caption"
        embedding_base_path = f"{args.data_path}/msvd_embeddings"
        base_chat_log_dir = f"{args.output_dir}/chatlog_rerank_{args.dataset}"
        total = 0
        correct = 0
        wrong = 0
        cannot_check = []
        rank_sum = [0, 0, 0, 0, 0, 0]
        top1_acc = 0
        top5_acc = 0
        top10_acc = 0 

        zs_top1_acc = 0
        zs_top5_acc = 0
        zs_top10_acc = 0

        vqa = Answerer(os.environ["OPENAI_API_KEY"])
        reranker = Reranker(
            project_id=os.environ["GOOGLE_CLOUD_PROJECT_ID"],
            location=os.environ["GOOGLE_CLOUD_LOCATION"],
            memory_path=embedding_base_path,
            queries=queries
        )

        # System prompt for the conversation
        conversation_system_prompt = {
            "role": "system",
            "content": """
            You are given caption about certain video(anchor video) and query used to retrieve the anchor video. However this video may not be the exact video the I am looking for. 
            Your role is to ask question about the video I have in mind to get more information about video. You have 3 rounds and you can only ask one question at a time.
            Don't just answer in yes or no. Answer concisely.
            Focus on attributes like number of people, color, shape.
            """
        }

        # iterate with query
        for row in predictions:
            record = {}
            rank_history = []
            
            total+=1
            
            target_vid = row["query_id"]
            topk = row["org_ranking"]
            
            logger.info(f"Processing target video: {target_vid}")
            logger.debug(f"Top-k videos: {topk}")
            
            if os.path.isfile(os.path.join(base_chat_log_dir, f'log_{target_vid}.json')):
                with open(os.path.join(base_chat_log_dir, f'log_{target_vid}.json'), 'r') as f:
                    pred = json.load(f)
                try:
                    r0 = pred['initial_rank']
                    r1 = pred['round1']['target_rank']
                    r2 = pred['round2']['target_rank']
                    r3 = pred['round3']['target_rank']
                    r4 = pred['round4']['target_rank']
                    r5 = pred['round5']['target_rank']
                    rank_history = [r0, r1, r2, r3, r4, r5]
                    logger.debug(f"Rank history: {rank_history}")
                    for _idx, r in enumerate(rank_history):
                        rank_sum[_idx]+=r
                        logger.info(f"Average ranking in round {_idx}: {round(rank_sum[_idx]/total, 1)} among {total} samples")

                    # topk
                    if r5==1:
                        top1_acc += 1
                    if r5<=5:
                        top5_acc += 1
                    if r5<=10:
                        top10_acc += 1
                    
                    ## zs retrieval ##
                    ranking = row["org_ranking"]
                    if target_vid == ranking[0]:
                        zs_top1_acc += 1
                    if target_vid in ranking[:5]:
                        zs_top5_acc += 1
                    if target_vid in ranking[:10]:
                        zs_top10_acc += 1

                    logger.info(f"Top-1 accuracy: {top1_acc / total * 100:.2f}% ({zs_top1_acc}->{top1_acc})/{total}")
                    logger.info(f"Top-5 accuracy: {top5_acc / total * 100:.2f}% ({zs_top5_acc}->{top5_acc})/{total}")
                    logger.info(f"Top-10 accuracy: {top10_acc / total * 100:.2f}% ({zs_top10_acc}->{top10_acc})/{total}")

                except Exception as e:
                    logger.error(f"Failed to process video {target_vid}: {str(e)}")
                    cannot_check.append(target_vid)
                    total-=1
                continue

            vqa.load_topk([os.path.join(video_base_path, f"{vid}.avi") for vid in topk])
            vqa.load_video(os.path.join(video_base_path, f"{target_vid}.avi"))
            
            anchor_captions = ""
            anchor = topk[0]
            for k in topk:
                try:
                    anchor_captions = video_captions[k]
                    break
                except:
                    pass

            reranker.init_embedding(target_vid)
            _, initial_rank = reranker.rerank(target_vid, video_embs)
            logger.info(f"Initial rank: {initial_rank}")
            if initial_rank==1:
                continue
            # record
            record['target_vid'] = target_vid
            record['topk_candidates'] = topk
            record['anchor_caption'] = anchor_captions
            record['initial_rank'] = int(initial_rank)  # numpy int object cannot be dumped to json object
            rank_history.append(initial_rank)

            messages = [conversation_system_prompt]
            simple_chat_prompt = str = (
                f"""
                This is caption of retrieved video. Read the video captions and ask some question to gain more information to help find out exact video.
                Some video may not have caption due to API error saying sorry I can't provide blah blah.
                Captions for video: {anchor_captions}\n

                Question: 
                """
            )

            messages.append({"role": "user", "content": simple_chat_prompt})
            model = "gpt-4o" # "gpt-4-1106-preview" # "gpt-3.5-turbo-16k"

            client = openai.OpenAI()
            completion = client.chat.completions.create(
                model=model, messages=messages, max_tokens=1500, stream=False, temperature=0.2
            )
            response = completion.choices[0].message.content
            for i in range(5):
                logger.info(f"Processing round {i+1}")
                logger.debug(f"Question: {completion.choices[0].message.content}")
                # record
                record[f'round{i+1}'] = {}
                record[f'round{i+1}']['question'] = completion.choices[0].message.content

                question = {"role":"assistant", "content": completion.choices[0].message.content}
                messages.append(question)
                answer, before_aggr = asyncio.run(vqa.async_ask(completion.choices[0].message.content))
                emb = reranker.get_image_video_text_embeddings(contextual_text=answer)
                reranker.add_embedding(emb.text_embedding)
                reranked_topk, target_rank = reranker.rerank(target_vid, video_embs)
                reranked_top1_caption = ""

                for k in topk:
                    try:
                        reranked_top1_caption = video_captions[k]
                        break
                    except:
                        pass
                
                # get new caption
                user_side_answer = f"""answer: {answer}
                Based on your answer, here's caption of reranked video.
                caption: {reranked_top1_caption}
                Keep asking.
                Question: 
                """

                # record
                record[f'round{i+1}']['answer'] = answer
                record[f'round{i+1}']['answer_before_aggr'] = before_aggr
                record[f'round{i+1}']['reranked_topk'] = reranked_topk
                record[f'round{i+1}']['target_rank'] = int(target_rank)
                rank_history.append(target_rank)
                logger.info(f"Answer: {answer}")
                logger.info(f"Target rank: {target_rank}")
                messages.append({"role": "user", "content": user_side_answer})

                if i!=2:
                    completion = client.chat.completions.create(
                        model=model, messages=messages, max_tokens=1500, stream=False, temperature=0.7
                        )
                    messages.append({"role":"assistant", "content": completion.choices[0].message.content})

            with open(os.path.join(base_chat_log_dir, f'log_{target_vid}.json'), 'w')as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            logger.debug(f"Rank history: {rank_history}")
            
            # avg rank
            for _idx, r in enumerate(rank_history):
                rank_sum[_idx]+=r
                logger.info(f"Average ranking in round {_idx}: {round(rank_sum[_idx]/total, 1)} among {total} samples")
            
            # topk
            if target_vid == reranked_topk[0]:
                top1_acc += 1
            if target_vid in reranked_topk[:5]:
                top5_acc += 1
            if target_vid in reranked_topk[:10]:
                top10_acc += 1
            
            ## zs retrieval ##
            ranking = row["org_ranking"]
            if target_vid == ranking[0]:
                zs_top1_acc += 1
            if target_vid in ranking[:5]:
                zs_top5_acc += 1
            if target_vid in ranking[:10]:
                zs_top10_acc += 1

            logger.info(f"Top-1 accuracy: {top1_acc / total * 100:.2f}% ({zs_top1_acc}->{top1_acc})/{total}")
            logger.info(f"Top-5 accuracy: {top5_acc / total * 100:.2f}% ({zs_top5_acc}->{top5_acc})/{total}")
            logger.info(f"Top-10 accuracy: {top10_acc / total * 100:.2f}% ({zs_top10_acc}->{top10_acc})/{total}")

            import pdb; pdb.set_trace()
    except Exception as e:
        logger.error(f"Error running MERLIN: {str(e)}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    # python run_merlin.py --dataset msvd --data_path /path/to/data
    main()
