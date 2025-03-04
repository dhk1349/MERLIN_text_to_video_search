import os
import json
import time
import asyncio
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import argparse
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_utils import prepare_data, DATASET_CONFIGS, DatasetPaths
from utils.video_utils import video_frame_generator
from utils.logger import logger, setup_logger
from utils.setup_directories import verify_structure, create_directory_structure
from utils.env_utils import load_env_variables, get_required_env, get_optional_env

from typing import Optional
from human_agent.answerer import Answerer
from merlin.reranker import Reranker
from merlin.questioner import Questioner

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
        default=".env.local",
        help="Path to .env file"
    )
    
    # Number of rounds for question-answering
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="Number of rounds for question-answering iteration"
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
        
        # Get dataset-specific configuration early
        dataset_config = DATASET_CONFIGS[args.dataset]
        
        # Create dataset paths
        dataset_paths = DatasetPaths.from_base_path(args.data_path, dataset_config)
        
        # Get video paths from dataset_paths
        video_base_paths = [str(path) for path in dataset_paths.get_video_paths()]
        video_ext = dataset_config.video_ext
        
        # Load dataset
        queries, video_captions, video_embs, text_embs = prepare_data(
            dataset=args.dataset,
            video_path=args.data_path,
            caption=os.path.join(args.data_path, "gpt4o_caption")
        )
        
        # Set up other paths
        video_caption_path = f"{args.data_path}/gpt4o_caption"
        embedding_base_path = f"{args.data_path}/"
        base_chat_log_dir = f"{args.output_dir}/chatlog_rerank_{args.dataset}"

        # Retrieve top-5 video candidates based on cosine similarity
        logger.info("Zero-shot retrieval evaluation...")
        top_k = 10
        predictions = []

        for i, query_text_emb in tqdm(enumerate(text_embs)):
            similarities = cosine_similarity([query_text_emb], video_embs)
            top_k_indices = np.argsort(-similarities[0])[:top_k]
            
            # Create a dictionary for the prediction
            prediction = {
                "query_id": queries[i]["video"].replace(dataset_config.video_ext, ""),
                "org_ranking": [queries[j]["video"].replace(dataset_config.video_ext, "") for j in top_k_indices.squeeze()]
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

        total = 0
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
            queries=queries,
            video_ext=video_ext
        )
        
        # Initialize the Questioner with the number of rounds from args
        questioner = Questioner(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
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
            
            # Load videos for VQA
            logger.info("Loading videos for VQA...")
            try:
                # Find target video path
                target_path = dataset_paths.find_video_path(target_vid)
                if target_path is None:
                    raise FileNotFoundError(f"Could not find target video {target_vid} in any of the video paths")
                
                # Load target video
                vqa.load_video(str(target_path))
                
                # Find and load top-k videos
                topk_video_paths = []
                for vid in topk:
                    video_path = dataset_paths.find_video_path(vid)
                    if video_path is not None:
                        topk_video_paths.append(str(video_path))
                    else:
                        # If video not found, try constructing path manually as fallback
                        for video_base_path in video_base_paths:
                            fallback_path = os.path.join(video_base_path, f"{vid}{video_ext}")
                            if os.path.exists(fallback_path):
                                topk_video_paths.append(fallback_path)
                                break
                
                # Load top-k videos
                if topk_video_paths:
                    vqa.load_topk(topk_video_paths)
                else:
                    logger.warning(f"No top-k videos found for {target_vid}")
            
            except Exception as e:
                logger.error(f"Error loading videos: {str(e)}")
                continue
            if target_path is None:
                raise FileNotFoundError(f"Could not find target video {target_vid} in any of the video paths")
            
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
            
            # On experiment, we did not use this condition
            # You can use this condition to skip the videos if initial rank is good enough
            # if initial_rank==1:
            #     total-=1  
            #     continue
            # if initial_rank < 10:
            #     total-=1
            #     continue

            # record
            record['target_vid'] = target_vid
            record['topk_candidates'] = topk
            record['anchor_caption'] = anchor_captions
            record['initial_rank'] = int(initial_rank)  # numpy int object cannot be dumped to json object
            rank_history.append(initial_rank)
            
            # Reset the questioner for a new conversation
            questioner.reset_conversation(
                target_video_id=target_vid
            )
            
            # Generate the first question using the Questioner
            question_result = questioner.generate_question(
                video_captions=anchor_captions
            )
            
            # Extract the question
            response = question_result["question"]
            
            for i in range(args.num_rounds):
                logger.info(f"Processing round {i+1} of {args.num_rounds}")
                logger.debug(f"Question: {response}")
                
                # record the question in our local record
                record[f'round{i+1}'] = {}
                record[f'round{i+1}']['question'] = response

                # Process the question and get an answer
                answer, before_aggr = asyncio.run(vqa.async_ask(response))
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
                
                # Record the answer in the questioner's conversation log
                questioner.record_answer(
                    answer=answer,
                    reranked_caption=reranked_top1_caption,
                    target_rank=target_rank,
                    reranked_topk=reranked_topk
                )

                # record in our local record
                record[f'round{i+1}']['answer'] = answer
                record[f'round{i+1}']['answer_before_aggr'] = before_aggr
                record[f'round{i+1}']['reranked_topk'] = reranked_topk
                record[f'round{i+1}']['target_rank'] = int(target_rank)
                rank_history.append(target_rank)
                logger.info(f"Answer: {answer}")
                logger.info(f"Target rank: {target_rank}")

                # Check if this is the last round based on the loop counter
                is_last_round = (i == args.num_rounds - 1)
                
                # Generate the next question if not the last round
                if not is_last_round:
                    logger.info(f"Generating question for round {i + 2}")
                    # Generate the next question using the Questioner
                    # No need to pass conversation history as it's maintained internally
                    question_result = questioner.generate_question(
                        video_captions=anchor_captions,
                        temperature=0.7  # Use higher temperature after first round
                    )
                    response = question_result["question"]
                else:
                    logger.info(f"Final round {i + 1} completed, no more questions will be generated")

            # Export the full conversation log from the questioner
            conversation_log = questioner.export_conversation_log()
            
            # Add round information to the conversation log
            conversation_log["num_rounds"] = args.num_rounds
            
            # Add round information to each conversation entry
            for i, conv in enumerate(conversation_log["conversations"]):
                conv["round"] = i + 1
            
            record['conversation_log'] = conversation_log
            
            # Create the output directory if it doesn't exist
            os.makedirs(base_chat_log_dir, exist_ok=True)
            
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

    except Exception as e:
        logger.error(f"Error running MERLIN: {str(e)}")
        if args.debug:
            raise
        sys.exit(1)
    import pdb; pdb.set_trace()
if __name__ == "__main__":
    # python run_merlin.py --dataset msvd --data_path /path/to/data
    main()
