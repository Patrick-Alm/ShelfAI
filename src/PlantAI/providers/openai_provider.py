import os
from typing import Optional
from openai import OpenAI

class OpenAIClient:
    """Simple OpenAI client for grocery products analysis"""

    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 800
    ):
        """
        Initialize OpenAI client for grocery products recommendations
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model_name: OpenAI model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or provide api_key parameter."
                )

        self.client = OpenAI(api_key=api_key)

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(self, prompt: str) -> str:
        """
        Generate grocery products recommendation from field analysis prompt
        
        Args:
            prompt: Field analysis prompt with detection results
            
        Returns:
            grocery products recommendation text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content or "No response generated"
            
        except Exception as e:
            return f"Error generating recommendations: {e}"