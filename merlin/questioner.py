import openai
from typing import List, Dict, Any, Optional
import logging
import time
import os

class Questioner:
    """
    A class responsible for generating questions about video content to help
    refine search results in the MERLIN system.
    
    This class maintains conversation history internally and can be reset when a new conversation begins.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", temperature: float = 0.2):
        """
        Initialize the Questioner with OpenAI API credentials.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
            model: OpenAI model to use for question generation
            temperature: Temperature parameter for question generation
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.default_temperature = temperature
        
        # Define the default system prompt
        self.default_system_prompt = {
            "role": "system",
            "content": """
            You are given caption about certain video(anchor video) and query used to retrieve the anchor video. However this video may not be the exact video the I am looking for. 
            Your role is to ask question about the video I have in mind to get more information about video. You have 3 rounds and you can only ask one question at a time.
            Don't just answer in yes or no. Answer concisely.
            Focus on attributes like number of people, color, shape.
            """
        }
        
        # Initialize conversation state
        self.messages = []
        self.system_prompt = self.default_system_prompt
        self.conversation_log = []
        self.current_video_captions = ""
        self.target_video_id = None
    
    def reset_conversation(self, target_video_id: Optional[str] = None):
        """
        Reset the conversation history to start a new conversation.
        
        Args:
            target_video_id: Optional ID of the target video for this conversation
        """
        self.messages = []
        self.system_prompt = self.default_system_prompt
        self.conversation_log = []
        self.current_video_captions = ""
        self.target_video_id = target_video_id
        self.logger.debug(f"Conversation history reset for video ID: {target_video_id}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of message dictionaries representing the conversation history
        """
        return self.messages.copy()
    
    def get_conversation_log(self) -> List[Dict[str, Any]]:
        """
        Get the structured conversation log with questions and answers.
        
        Returns:
            List of dictionaries containing question-answer pairs and metadata
        """
        return self.conversation_log.copy()
    
    def add_to_conversation(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ("system", "user", or "assistant")
            content: The content of the message
        """
        message = {"role": role, "content": content}
        self.messages.append(message)
        self.logger.debug(f"Added {role} message to conversation history")
    
    def generate_question(self, 
                     video_captions: str,
                     conversation_history: Optional[List[Dict[str, str]]] = None,
                     max_tokens: int = 1500,
                     temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate a question about the video based on captions and conversation history.
        
        Args:
            video_captions: Captions describing the video content
            conversation_history: Optional conversation history to consider
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation (uses default if None)
            
        Returns:
            Dictionary containing the generated question and metadata
        """
        # Store the current video captions
        self.current_video_captions = video_captions
        
        # Use provided conversation history or the internal one
        if conversation_history is not None:
            # Reset the conversation if a new history is provided
            self.messages = []
            
        # Initialize messages with system prompt if empty
        if not self.messages:
            self.messages = [self.system_prompt]
        
        # Add the video captions as a user message
        user_message = f"""
        This is caption of retrieved video. Read the video captions and ask some question to gain more information to help find out exact video.
        Some video may not have caption due to API error saying sorry I can't provide blah blah.
        Captions for video: {video_captions}

        Question: 
        """
        
        self.add_to_conversation("user", user_message)
        
        # Generate the question using the OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=temperature or self.default_temperature
            )
            
            # Extract the question from the response
            question = response.choices[0].message.content.strip()
            
            # Add the assistant's response to the conversation
            self.add_to_conversation("assistant", question)
            
            # Return the question and metadata
            return {
                "question": question,
                "model": self.model,
                "temperature": temperature or self.default_temperature,
                "max_tokens": max_tokens,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating question: {str(e)}")
            return {
                "question": "Could not generate a question due to an error.",
                "error": str(e)
            }
    
    def record_answer(self, answer: str, reranked_caption: str, target_rank: Optional[int] = None, reranked_topk: Optional[List[str]] = None):
        """
        Record an answer to the most recent question in the conversation log.
        
        Args:
            answer: The answer to the question
            reranked_caption: Caption of the reranked top video
            target_rank: The rank of the target video after reranking (optional)
            reranked_topk: List of top-k reranked video IDs (optional)
        """
        if self.conversation_log:
            # Update the most recent entry in the conversation log
            self.conversation_log[-1]["answer"] = answer
            self.conversation_log[-1]["reranked_caption"] = reranked_caption
            self.conversation_log[-1]["answer_timestamp"] = time.time()
            
            if target_rank is not None:
                self.conversation_log[-1]["target_rank"] = target_rank
            
            if reranked_topk is not None:
                self.conversation_log[-1]["reranked_topk"] = reranked_topk
            
            # Add the answer to the conversation history
            formatted_answer = self.format_answer_prompt(answer, reranked_caption)
            self.add_to_conversation("user", formatted_answer)
            
            self.logger.debug(f"Recorded answer, target rank: {target_rank}")
        else:
            self.logger.warning("Attempted to record answer but no questions exist in conversation log")
    
    def format_answer_prompt(self, answer: str, reranked_caption: str) -> str:
        """
        Format the prompt for the next question based on the answer and reranked video caption.
        
        Args:
            answer: The answer to the previous question
            reranked_caption: Caption of the reranked top video
            
        Returns:
            Formatted prompt string
        """
        return f"""answer: {answer}
        Based on your answer, here's caption of reranked video.
        caption: {reranked_caption}
        Keep asking.
        Question: 
        """
    
    def export_conversation_log(self) -> Dict[str, Any]:
        """
        Export the full conversation log in a structured format.
        
        Returns:
            Dictionary containing the full conversation history and metadata
        """
        return {
            "target_video_id": self.target_video_id,
            "conversations": self.conversation_log,
            "total_conversations": len(self.conversation_log),
            "system_prompt": self.system_prompt["content"],
            "timestamp": time.time()
        } 