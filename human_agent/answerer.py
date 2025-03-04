import torch
import numpy as np
import aiohttp
import asyncio
import base64
import io
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import OpenAI

from utils.video_utils import video_frame_generator

class Answerer(torch.nn.Module):
    def __init__(self, api_key):
        super(Answerer, self).__init__()
        self.api_key = api_key
        # Load VQA model
        # config_path = './mPLUG/configs_video/VideoQA_msrvtt_large.yaml'
        # checkpoint_path = './mPLUG2_MSRVTT_QA.pth'
        # video_qa_model = VideoQAModel(config_path, checkpoint_path)

        # Load VQA model-> BLIP2 + ChatGPT3.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vqa_model = "gpt-4o"
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
To apply it to videos, frames are uniformly extracted from the video over time, and the model provides an answer for each frame to a given question. This means that for a single question, there will be multiple answers - one for each extracted frame. Your role is to review all of the individual answers and summarize them to provide a final answer to the original question. When making final answer, don't user unnecessary words like 'Based on the individual answers provided by the VQA model,'. Just answer to the question.

For example, if the question is "Did a cookie appear in the video?" and the individual answers from the frames are ["No", "No", "Yes", "No"],  then since a cookie appeared in the 3rd frame, you should summarize and answer the question as "Yes". Length of aggregated answer should be around 30~35 words.""",
        } 
        
        self.model = "gpt-4o" # "gpt-4-1106-preview" "gpt-3.5-turbo-16k"
        self.client = OpenAI(api_key=self.api_key)
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
        # for image in images:
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
                print("Answer failed..")   # request error
                
            if answers!=None:
                ans, perplexity = self.aggregate(question, answers)
            else: 
                ans = "no answer available"
                answers = ["no answer available", "no answer available", "no answer available"]
                perplexity = 9
        # we did not used perplexity in the experiment
        return ans, answers #, perplexity 
    
    def aggregate(self, question, answers):
        aggregation_prompt = {"role": "user", "content": f""" 
            Question: {question}\n
            VQA answers: {answers}\n
            Agregated Answer: 
            """}
        messages = [self.system_prompt, aggregation_prompt]
        
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=100, stream=False, temperature=0.5,
            logprobs=True
        )
        response = completion.choices[0].message.content
        logprobs = [item.logprob for item in completion.choices[0].logprobs.content]
        perplexity = np.exp(-np.sum(logprobs) / len(logprobs))
        return response, perplexity