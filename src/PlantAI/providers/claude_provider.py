
import os
from typing import Optional
from anthropic import Anthropic


class ClaudeClient:
    """Simple Claude client for grocery products analysis"""
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: str = "claude-3-5-sonnet-20241022", 
        temperature: float = 0.9,
        max_output_tokens: int = 8192,
        top_p: float = 1.0,
        top_k: int = 32
    ):
        """
        Initialize Claude client for grocery products recommendations
        
        Args:
            api_key: Claude API key (or set ANTHROPIC_API_KEY env var)
            model_name: Claude model to use
        """
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Claude API key not found. Set ANTHROPIC_API_KEY environment variable "
                    "or provide api_key parameter."
                )

        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def generate_response(self, prompt: str) -> str:
        """
        Generate grocery products recommendation from field analysis prompt
        
        Args:
            prompt: Field analysis prompt with detection results
        
        Returns:
            str: The generated recommendation text or error message.
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Handle different content block types
            result_text = ""
            for content_block in response.content:
                if content_block.type == "text":
                    result_text += content_block.text

            return result_text if result_text else "No text response generated"

        except Exception as e:
            return f"Error generating recommendations: {e}"