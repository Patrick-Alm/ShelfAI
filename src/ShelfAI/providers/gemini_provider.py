import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from typing import Optional


class GeminiClient:
    """Simple Gemini client for retail shelf analysis"""

    def _init_(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        max_output_tokens: int = 800
    ):
        """
        Initialize Gemini client for retail recommendations
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            temperature: Temperature for response generation
            max_output_tokens: Maximum tokens to generate
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Google API key not found. Set GOOGLE_API_KEY environment variable "
                    "or provide api_key parameter."
                )

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        )

    def generate_response(self, prompt: str) -> str:
        """
        Generate retail recommendation from shelf analysis prompt
        
        Args:
            prompt: Shelf analysis prompt with detection results
            
        Returns:
            Retail recommendation text
        """
        try:
            response = self.model.generate_content(prompt)
            
            if hasattr(response, "text"):
                return response.text
            elif response.candidates and response.candidates[0].content.parts:
                first_part = response.candidates[0].content.parts[0]
                return first_part.text if hasattr(first_part, "text") else str(first_part)
            else:
                return "No response generated"
                
        except Exception as e:
            return f"Error generating recommendations: {e}"
