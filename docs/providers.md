# Provider System Documentation

The ShelfAI provider system offers a unified interface for multiple Large Language Model (LLM) providers, enabling flexible AI-powered shelf analysis with Claude, Gemini, and OpenAI models.

## Overview

The provider architecture implements a consistent interface across different LLM services, allowing seamless switching between providers while maintaining the same functionality and output format.

## Supported Providers

### Claude (Anthropic)
- **Model**: `claude-3-5-sonnet-20241022`
- **Strengths**: High-quality analysis, detailed reasoning, retail expertise
- **Best for**: Complex shelf analysis, detailed recommendations
- **API**: Anthropic Messages API

### Gemini (Google)  
- **Model**: `gemini-1.5-flash`
- **Strengths**: Fast responses, efficient processing, good accuracy
- **Best for**: Real-time analysis, batch processing, cost optimization
- **API**: Google Generative AI

### OpenAI
- **Model**: `gpt-4o-mini`
- **Strengths**: Reliable performance, consistent outputs, wide adoption
- **Best for**: Standard retail analysis, integration with existing OpenAI workflows
- **API**: OpenAI Chat Completions

## Provider Interface

All providers implement the same unified interface:

```python
class BaseProvider:
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize provider with API key and configuration"""
        pass
    
    def generate_response(self, prompt: str) -> str:
        """Generate analysis response from prompt"""
        pass
```

## Individual Provider Documentation

### ClaudeClient

```python
from PlantAI.providers import ClaudeClient

client = ClaudeClient(
    api_key="your-api-key",           # Optional, uses ANTHROPIC_API_KEY env var
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.9,                  # High creativity for varied insights
    max_output_tokens=8192,           # Detailed analysis support
    top_p=1.0,
    top_k=32
)

response = client.generate_response("Analyze this shelf data...")
```

**Configuration Options:**
- `model_name`: Claude model version
- `temperature`: 0.0-1.0, controls response creativity
- `max_output_tokens`: Maximum response length (up to 8192)
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter

**Features:**
- Detailed analytical responses
- Strong reasoning capabilities
- Excellent for complex retail scenarios
- Handles long, detailed prompts well

### GeminiClient

```python
from PlantAI.providers import GeminiClient

client = GeminiClient(
    api_key="your-api-key",           # Optional, uses GOOGLE_API_KEY env var
    model_name="gemini-1.5-flash",
    temperature=0.3,                  # Balanced creativity
    max_output_tokens=800             # Efficient response length
)

response = client.generate_response("Analyze this shelf data...")
```

**Configuration Options:**
- `model_name`: Gemini model variant
- `temperature`: 0.0-1.0, response creativity
- `max_output_tokens`: Maximum response length (up to 2048)

**Features:**
- Fast response times
- Efficient token usage
- Good for real-time applications
- Reliable structured outputs

### OpenAIClient

```python
from PlantAI.providers import OpenAIClient

client = OpenAIClient(
    api_key="your-api-key",           # Optional, uses OPENAI_API_KEY env var
    model_name="gpt-4o-mini",
    temperature=0.3,                  # Consistent outputs
    max_tokens=800                    # Standard response length
)

response = client.generate_response("Analyze this shelf data...")
```

**Configuration Options:**
- `model_name`: OpenAI model variant
- `temperature`: 0.0-2.0, response creativity
- `max_tokens`: Maximum response length

**Features:**
- Consistent, reliable outputs
- Wide model selection
- Good general-purpose analysis
- Excellent for standard retail insights

## Usage Examples

### Provider Comparison

```python
from PlantAI.providers import ClaudeClient, GeminiClient, OpenAIClient

# Initialize all providers
providers = {
    'claude': ClaudeClient(),
    'gemini': GeminiClient(), 
    'openai': OpenAIClient()
}

prompt = """
Analyze this grocery shelf:
- 25 products detected
- 60% food items, 30% beverages, 10% household
- High shelf density
- Grocery section

Provide inventory status and recommendations.
"""

# Compare responses
for name, client in providers.items():
    response = client.generate_response(prompt)
    print(f"\n=== {name.upper()} ===")
    print(response)
```

### Performance Testing

```python
import time

def benchmark_providers(prompt, iterations=5):
    providers = {
        'claude': ClaudeClient(temperature=0.1),
        'gemini': GeminiClient(temperature=0.1),
        'openai': OpenAIClient(temperature=0.1)
    }
    
    results = {}
    
    for name, client in providers.items():
        times = []
        for i in range(iterations):
            start = time.time()
            response = client.generate_response(prompt)
            times.append(time.time() - start)
        
        results[name] = {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times)
        }
    
    return results

# Run performance test
benchmark_results = benchmark_providers("Analyze shelf with 15 products...")
for provider, metrics in benchmark_results.items():
    print(f"{provider}: {metrics['avg_time']:.2f}s average")
```

### Cost Optimization

```python
class CostOptimizedAnalysis:
    def __init__(self):
        # Use most cost-effective provider first
        self.primary = GeminiClient()      # Fastest, cheapest
        self.secondary = OpenAIClient()    # Reliable fallback
        self.premium = ClaudeClient()      # High-quality analysis
    
    def analyze_with_fallback(self, prompt):
        # Try providers in cost-effectiveness order
        try:
            return self.primary.generate_response(prompt)
        except Exception:
            try:
                return self.secondary.generate_response(prompt)
            except Exception:
                return self.premium.generate_response(prompt)
    
    def analyze_by_complexity(self, prompt, complexity='medium'):
        # Choose provider based on analysis complexity
        if complexity == 'simple':
            return self.primary.generate_response(prompt)
        elif complexity == 'medium':
            return self.secondary.generate_response(prompt)
        else:  # complex
            return self.premium.generate_response(prompt)
```

## Advanced Configuration

### Custom Provider Implementation

```python
from PlantAI.providers.base import BaseProvider
import requests

class CustomLLMProvider(BaseProvider):
    def __init__(self, api_endpoint, api_key, **kwargs):
        self.endpoint = api_endpoint
        self.api_key = api_key
        self.config = kwargs
    
    def generate_response(self, prompt: str) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'prompt': prompt,
            'max_tokens': self.config.get('max_tokens', 800),
            'temperature': self.config.get('temperature', 0.3)
        }
        
        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        
        return response.json()['text']

# Usage
custom_provider = CustomLLMProvider(
    api_endpoint='https://api.example.com/generate',
    api_key='your-key',
    max_tokens=1000,
    temperature=0.5
)
```

### Provider Pool Management

```python
import random
from concurrent.futures import ThreadPoolExecutor

class ProviderPool:
    def __init__(self):
        self.providers = {
            'claude': ClaudeClient(),
            'gemini': GeminiClient(),
            'openai': OpenAIClient()
        }
        self.weights = {
            'claude': 0.3,    # 30% chance
            'gemini': 0.4,    # 40% chance  
            'openai': 0.3     # 30% chance
        }
    
    def get_random_provider(self):
        """Get provider based on weighted random selection"""
        providers = list(self.providers.keys())
        weights = [self.weights[p] for p in providers]
        chosen = random.choices(providers, weights=weights)[0]
        return self.providers[chosen]
    
    def parallel_analysis(self, prompt, num_providers=2):
        """Run analysis on multiple providers in parallel"""
        selected_providers = random.sample(
            list(self.providers.values()), 
            num_providers
        )
        
        with ThreadPoolExecutor(max_workers=num_providers) as executor:
            futures = [
                executor.submit(provider.generate_response, prompt)
                for provider in selected_providers
            ]
            
            results = []
            for future in futures:
                try:
                    results.append(future.result(timeout=30))
                except Exception as e:
                    results.append(f"Error: {e}")
            
            return results
    
    def consensus_analysis(self, prompt):
        """Get consensus from multiple providers"""
        results = self.parallel_analysis(prompt, num_providers=3)
        
        # Simple consensus: return most common recommendation
        recommendations = []
        for result in results:
            if 'high priority' in result.lower():
                recommendations.append('high')
            elif 'low priority' in result.lower():
                recommendations.append('low')
            else:
                recommendations.append('medium')
        
        # Return most frequent recommendation
        return max(set(recommendations), key=recommendations.count)
```

## Error Handling

### Robust Error Management

```python
import time
import logging

class RobustProvider:
    def __init__(self, provider_name='claude'):
        self.providers = {
            'claude': ClaudeClient(),
            'gemini': GeminiClient(),
            'openai': OpenAIClient()
        }
        self.primary = self.providers[provider_name]
        self.fallbacks = [p for name, p in self.providers.items() if name != provider_name]
    
    def generate_response_with_retry(self, prompt, max_retries=3):
        """Generate response with automatic retry and fallback"""
        
        # Try primary provider
        for attempt in range(max_retries):
            try:
                return self.primary.generate_response(prompt)
            except Exception as e:
                logging.warning(f"Primary provider attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                else:
                    logging.error("Primary provider failed, trying fallbacks")
        
        # Try fallback providers
        for i, fallback in enumerate(self.fallbacks):
            try:
                logging.info(f"Trying fallback provider {i + 1}")
                return fallback.generate_response(prompt)
            except Exception as e:
                logging.warning(f"Fallback {i + 1} failed: {e}")
        
        # All providers failed
        raise Exception("All providers failed to generate response")

# Usage
robust_analyzer = RobustProvider('gemini')
try:
    response = robust_analyzer.generate_response_with_retry(prompt)
except Exception as e:
    print(f"Analysis failed: {e}")
```

### API Key Management

```python
import os
from typing import Optional

class SecureProviderManager:
    @staticmethod
    def get_api_key(provider: str, custom_key: Optional[str] = None) -> str:
        """Secure API key retrieval with multiple fallback methods"""
        
        if custom_key:
            return custom_key
        
        # Environment variable mapping
        env_vars = {
            'claude': 'ANTHROPIC_API_KEY',
            'gemini': 'GOOGLE_API_KEY',
            'openai': 'OPENAI_API_KEY'
        }
        
        env_key = os.getenv(env_vars.get(provider))
        if env_key:
            return env_key
        
        # Try alternative environment variable names
        alt_vars = {
            'claude': ['CLAUDE_API_KEY', 'ANTHROPIC_KEY'],
            'gemini': ['GEMINI_API_KEY', 'GOOGLE_AI_KEY'],
            'openai': ['OPENAI_KEY', 'GPT_API_KEY']
        }
        
        for alt_var in alt_vars.get(provider, []):
            alt_key = os.getenv(alt_var)
            if alt_key:
                return alt_key
        
        raise ValueError(f"No API key found for {provider}")
    
    @classmethod
    def create_provider(cls, provider_name: str, api_key: Optional[str] = None, **kwargs):
        """Factory method for secure provider creation"""
        secure_key = cls.get_api_key(provider_name, api_key)
        
        providers = {
            'claude': ClaudeClient,
            'gemini': GeminiClient,
            'openai': OpenAIClient
        }
        
        provider_class = providers.get(provider_name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        return provider_class(api_key=secure_key, **kwargs)
```

## Best Practices

### Provider Selection Guidelines

1. **Claude**: Use for complex analysis requiring detailed reasoning
2. **Gemini**: Use for fast, efficient analysis and real-time applications  
3. **OpenAI**: Use for reliable, consistent standard analysis

### Performance Optimization

```python
# Temperature settings for different use cases
TEMPERATURE_CONFIGS = {
    'consistent': 0.1,      # Minimal variation, reliable outputs
    'balanced': 0.3,        # Good mix of consistency and creativity
    'creative': 0.7,        # More varied insights and recommendations
    'experimental': 1.0     # Maximum creativity for exploring options
}

# Token limits for cost optimization
TOKEN_LIMITS = {
    'summary': 200,         # Brief insights only
    'standard': 800,        # Comprehensive analysis
    'detailed': 2000,       # In-depth recommendations
    'comprehensive': 4000   # Full detailed report
}
```

### Error Prevention

```python
def validate_prompt(prompt: str) -> bool:
    """Validate prompt before sending to provider"""
    if len(prompt) < 10:
        raise ValueError("Prompt too short")
    if len(prompt) > 50000:
        raise ValueError("Prompt too long")
    if not any(word in prompt.lower() for word in ['shelf', 'product', 'inventory']):
        raise ValueError("Prompt must be related to shelf analysis")
    return True

def sanitize_response(response: str) -> str:
    """Clean and validate provider response"""
    if not response or len(response) < 10:
        return "Unable to generate meaningful analysis"
    
    # Remove potential sensitive information
    sensitive_patterns = ['api_key', 'password', 'token']
    for pattern in sensitive_patterns:
        if pattern in response.lower():
            response = response.replace(pattern, '[REDACTED]')
    
    return response.strip()
```

## Monitoring and Analytics

### Usage Tracking

```python
import time
from collections import defaultdict

class ProviderAnalytics:
    def __init__(self):
        self.usage_stats = defaultdict(list)
        self.error_counts = defaultdict(int)
    
    def track_usage(self, provider_name: str, response_time: float, success: bool):
        """Track provider usage statistics"""
        self.usage_stats[provider_name].append({
            'timestamp': time.time(),
            'response_time': response_time,
            'success': success
        })
        
        if not success:
            self.error_counts[provider_name] += 1
    
    def get_stats(self, provider_name: str = None):
        """Get usage statistics"""
        if provider_name:
            stats = self.usage_stats[provider_name]
            response_times = [s['response_time'] for s in stats if s['success']]
            
            return {
                'total_requests': len(stats),
                'success_rate': sum(s['success'] for s in stats) / len(stats),
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'error_count': self.error_counts[provider_name]
            }
        else:
            return {name: self.get_stats(name) for name in self.usage_stats.keys()}

# Usage with analytics
analytics = ProviderAnalytics()

def analyze_with_tracking(provider, prompt):
    start_time = time.time()
    try:
        response = provider.generate_response(prompt)
        success = True
    except Exception as e:
        response = f"Error: {e}"
        success = False
    
    response_time = time.time() - start_time
    analytics.track_usage(provider.__class__.__name__, response_time, success)
    
    return response
```

This comprehensive provider system enables flexible, robust, and scalable LLM integration for shelf analysis while maintaining consistent interfaces and comprehensive error handling.# Provider System Documentation

The ShelfAI provider system offers a unified interface for multiple Large Language Model (LLM) providers, enabling flexible AI-powered shelf analysis with Claude, Gemini, and OpenAI models.

## Overview

The provider architecture implements a consistent interface across different LLM services, allowing seamless switching between providers while maintaining the same functionality and output format.

## Supported Providers

### Claude (Anthropic)
- **Model**: `claude-3-5-sonnet-20241022`
- **Strengths**: High-quality analysis, detailed reasoning, retail expertise
- **Best for**: Complex shelf analysis, detailed recommendations
- **API**: Anthropic Messages API

### Gemini (Google)  
- **Model**: `gemini-1.5-flash`
- **Strengths**: Fast responses, efficient processing, good accuracy
- **Best for**: Real-time analysis, batch processing, cost optimization
- **API**: Google Generative AI

### OpenAI
- **Model**: `gpt-4o-mini`
- **Strengths**: Reliable performance, consistent outputs, wide adoption
- **Best for**: Standard retail analysis, integration with existing OpenAI workflows
- **API**: OpenAI Chat Completions

## Provider Interface

All providers implement the same unified interface:

```python
class BaseProvider:
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize provider with API key and configuration"""
        pass
    
    def generate_response(self, prompt: str) -> str:
        """Generate analysis response from prompt"""
        pass
```

## Individual Provider Documentation