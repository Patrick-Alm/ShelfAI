# ShelfAnalyzer Documentation

The `ShelfAnalyzer` class integrates computer vision detection with large language model analysis to provide comprehensive retail shelf insights and actionable recommendations for store management.

## Overview

ShelfAnalyzer combines ProductDetector results with AI-powered analysis to deliver retail-specific insights including inventory assessment, product mix evaluation, restocking priorities, and customer experience optimization.

## Features

- **Multi-Provider LLM Support**: Claude, Gemini, and OpenAI integration
- **Retail-Focused Analysis**: Specialized prompts for grocery/retail environments
- **Actionable Insights**: Practical recommendations for store operations
- **Automated Parsing**: Intelligent extraction of key metrics from AI responses
- **Flexible Output**: Summary reports, detailed analysis, and custom formats
- **Error Handling**: Graceful fallbacks for API failures

## Class Reference

### Constructor

```python
ShelfAnalyzer(provider="claude", api_key=None, **kwargs)
```

**Parameters:**
- `provider` (str): LLM provider ('claude', 'gemini', 'openai'). Default: 'claude'
- `api_key` (str, optional): API key for the provider. Uses environment variables if None.
- `**kwargs`: Additional provider-specific parameters

**Provider Configuration:**
```python
# Claude configuration
analyzer = ShelfAnalyzer(
    provider='claude',
    model_name='claude-3-5-sonnet-20241022',
    temperature=0.9,
    max_output_tokens=8192
)

# Gemini configuration  
analyzer = ShelfAnalyzer(
    provider='gemini',
    model_name='gemini-1.5-flash',
    temperature=0.3,
    max_output_tokens=800
)

# OpenAI configuration
analyzer = ShelfAnalyzer(
    provider='openai',
    model_name='gpt-4o-mini',
    temperature=0.3,
    max_tokens=800
)
```

### Methods

#### `analyze_shelf(image_path: str, store_section: str = "grocery") -> Dict`

Comprehensive shelf analysis combining detection and AI insights.

**Parameters:**
- `image_path` (str): Path to the shelf image
- `store_section` (str): Store section type for contextual analysis

**Returns:**
Dictionary containing:
```python
{
    'detection_results': Dict,  # Full ProductDetector results
    'insights': Dict,          # Parsed AI recommendations
    'summary': str             # Formatted summary report
}
```

**Insights Structure:**
```python
{
    'full_insights': str,           # Complete AI analysis
    'inventory_status': str,        # Well-stocked/Understocked/Overstocked
    'product_mix': str,            # Variety assessment
    'restocking_priority': str,     # high/medium/low
    'organization': str,           # Layout recommendations
    'action_steps': List[str]      # Specific action items
}
```

#### `quick_analysis(image_path: str, store_section: str = "grocery") -> str`

Fast analysis returning only the summary report.

**Parameters:**
- `image_path` (str): Path to the shelf image
- `store_section` (str): Store section type

**Returns:**
- `str`: Formatted summary with key insights and recommendations

#### `save_results(analysis_result: Dict, output_file: str = "shelf_analysis.txt")`

Saves analysis results to a text file.

**Parameters:**
- `analysis_result` (Dict): Result from `analyze_shelf()`
- `output_file` (str): Output file path

## Usage Examples

### Basic Analysis

```python
from ShelfAI import ShelfAnalyzer

# Initialize with default Claude provider
analyzer = ShelfAnalyzer()

# Run complete analysis
results = analyzer.analyze_shelf('shelf_image.jpg')

print("Summary:")
print(results['summary'])

print("\nInventory Status:")
print(results['insights']['inventory_status'])

print("\nAction Items:")
for action in results['insights']['action_steps']:
    print(f"• {action}")
```

### Provider Comparison

```python
from ShelfAI import ShelfAnalyzer

# Compare insights across providers
providers = ['claude', 'gemini', 'openai']
image_path = 'grocery_shelf.jpg'

for provider in providers:
    analyzer = ShelfAnalyzer(provider=provider)
    summary = analyzer.quick_analysis(image_path)
    print(f"\n=== {provider.upper()} ANALYSIS ===")
    print(summary)
```

### Store Section Analysis

```python
# Analyze different store sections
sections = ['grocery', 'beverages', 'household', 'pharmacy']
analyzer = ShelfAnalyzer(provider='claude')

for section in sections:
    results = analyzer.analyze_shelf(f'{section}_shelf.jpg', section)
    print(f"\n{section.upper()} SECTION:")
    print(f"Priority: {results['insights']['restocking_priority']}")
    print(f"Status: {results['insights']['inventory_status']}")
```

### Detailed Analysis Workflow

```python
from ShelfAI import ShelfAnalyzer
import json

analyzer = ShelfAnalyzer(provider='gemini', temperature=0.1)

# Run comprehensive analysis
results = analyzer.analyze_shelf('retail_shelf.jpg', 'grocery')

# Extract key metrics
detection = results['detection_results']
insights = results['insights']

# Decision logic based on analysis
if insights['restocking_priority'] == 'high':
    print("🚨 URGENT: Immediate restocking required")
    
if detection['shelf_density'] == 'very_low':
    print("📉 ALERT: Shelf appears nearly empty")
    
if insights['inventory_status'] == 'Overstocked':
    print("📊 NOTICE: Consider reducing order quantities")

# Save detailed report
analyzer.save_results(results, 'detailed_shelf_report.txt')

# Export as JSON for systems integration
with open('shelf_data.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Analysis Components

### Inventory Assessment

The analyzer evaluates stock levels based on:
- **Product density**: Total items per shelf area
- **Category distribution**: Balance of food/beverages/household items
- **Visual gaps**: Empty shelf spaces indicating low stock

**Status Classifications:**
- `Well-stocked`: Optimal product density and variety
- `Understocked`: Below optimal levels, restocking needed
- `Overstocked`: Excessive inventory, may impact sales

### Product Mix Analysis

Evaluates product variety and category balance:
- **Diversity Score**: Range of product categories
- **Category Balance**: Appropriate mix for store section
- **Missing Categories**: Gaps in expected product types

**Mix Classifications:**
- `Good product variety`: Balanced, diverse selection
- `Limited variety`: Narrow product range
- `Needs diversity`: Missing key categories

### Restocking Priority

AI-driven priority assessment:
- **High**: Immediate action required (empty sections, critical shortages)
- **Medium**: Plan restocking within 1-2 days
- **Low**: Monitor, restock during regular schedule

### Organization Recommendations

Layout and presentation suggestions:
- **Needs reorganization**: Poor product arrangement
- **Optimize placement**: Better positioning for high-demand items
- **Improve signage**: Better category labeling needed
- **Review full insights**: Complex recommendations requiring detailed review

## Prompt Engineering

### Standard Analysis Prompt

The analyzer uses specialized prompts for retail analysis:

```python
prompt = f"""
Supermarket Shelf Analysis Report:

Store Section: {store_section}
Total products detected: {detection_results['total_detections']}
Food items: {detection_results['food_items']} 
Beverages: {detection_results['beverages']}
Household items: {detection_results['household_items']}
Other products: {detection_results['other_products']}
Food percentage: {detection_results['food_percentage']}%
Shelf density: {detection_results['shelf_density']}

Please provide retail insights:
1. Inventory status (well-stocked, understocked, overstocked)
2. Product mix analysis (good variety, needs diversity, etc.)
3. Shelf organization recommendations
4. Restocking priority (high, medium, low)
5. Customer experience impact
6. Specific action items for store management

Keep insights practical and actionable for retail operations.
"""
```

### Custom Prompts

```python
class CustomShelfAnalyzer(ShelfAnalyzer):
    def _get_retail_insights(self, detection_results, store_section):
        # Custom prompt for specific business needs
        custom_prompt = f"""
        Analyze this {store_section} shelf for our premium grocery chain:
        
        Metrics: {detection_results}
        
        Focus on:
        - Premium product placement
        - Customer flow optimization  
        - Revenue maximization strategies
        - Competitive positioning
        """
        
        return self.client.generate_response(custom_prompt)
```

## Advanced Features

### Insight Parsing

The analyzer automatically extracts key information from AI responses:

```python
def _parse_insights(self, insights: str) -> Dict:
    parsed = {}
    lines = insights.lower().split('\n')

    # Pattern matching for key insights
    patterns = {
        'inventory': ['well-stocked', 'understocked', 'overstocked'],
        'priority': ['high priority', 'low priority', 'urgent'],
        'mix': ['good variety', 'poor variety', 'diverse'],
        'organization': ['reorganize', 'rearrange', 'optimize']
    }
    
    # Extract structured data from natural language
    for line in lines:
        for category, keywords in patterns.items():
            for keyword in keywords:
                if keyword in line:
                    parsed[category] = self._classify_insight(keyword)
                    
    return parsed
```

### Error Handling

Robust error handling for API failures:

```python
try:
    advice = self.client.generate_response(prompt)
    parsed = self._parse_insights(advice)
    return structured_insights
except Exception as e:
    return {
        'full_insights': f"Error: {e}",
        'inventory_status': 'Manual assessment needed',
        'product_mix': 'Consult store manager',
        'restocking_priority': 'unknown',
        'organization': 'Review manually',
        'action_steps': ['Consult store management']
    }
```

## Integration Examples

### Automated Monitoring System

```python
import schedule
import time
from datetime import datetime

def daily_shelf_check():
    analyzer = ShelfAnalyzer(provider='claude')
    
    # Check multiple shelves
    shelves = ['aisle1.jpg', 'aisle2.jpg', 'aisle3.jpg']
    alerts = []
    
    for shelf in shelves:
        results = analyzer.analyze_shelf(shelf)
        if results['insights']['restocking_priority'] == 'high':
            alerts.append(f"HIGH PRIORITY: {shelf}")
    
    if alerts:
        send_management_alert(alerts)

# Schedule daily checks
schedule.every().day.at("08:00").do(daily_shelf_check)
```

### Inventory Management Integration

```python
class InventorySystem:
    def __init__(self):
        self.analyzer = ShelfAnalyzer(provider='gemini')
        
    def generate_restocking_orders(self, shelf_images):
        orders = {}
        
        for shelf_id, image_path in shelf_images.items():
            analysis = self.analyzer.analyze_shelf(image_path)
            
            if analysis['insights']['restocking_priority'] == 'high':
                # Calculate reorder quantities based on detected gaps
                detection = analysis['detection_results']
                orders[shelf_id] = self._calculate_reorder(detection)
                
        return orders
    
    def _calculate_reorder(self, detection_results):
        # Business logic for reorder quantities
        if detection_results['shelf_density'] == 'very_low':
            return {'quantity': 'full_restock', 'urgency': 'immediate'}
        elif detection_results['shelf_density'] == 'low':
            return {'quantity': 'partial_restock', 'urgency': 'within_24h'}
        return {'quantity': 'maintenance', 'urgency': 'normal'}
```

### Business Intelligence Dashboard

```python
import pandas as pd

class ShelfAnalyticsDashboard:
    def __init__(self):
        self.analyzer = ShelfAnalyzer(provider='openai')
        
    def generate_weekly_report(self, shelf_data):
        results = []
        
        for date, shelves in shelf_data.items():
            for shelf_id, image_path in shelves.items():
                analysis = self.analyzer.analyze_shelf(image_path)
                
                results.append({
                    'date': date,
                    'shelf_id': shelf_id,
                    'total_products': analysis['detection_results']['total_detections'],
                    'shelf_density': analysis['detection_results']['shelf_density'],
                    'inventory_status': analysis['insights']['inventory_status'],
                    'priority': analysis['insights']['restocking_priority']
                })
        
        df = pd.DataFrame(results)
        return self._create_insights_dashboard(df)
    
    def _create_insights_dashboard(self, df):
        # Generate business insights from aggregated data
        trends = {
            'avg_products_per_shelf': df['total_products'].mean(),
            'high_priority_shelves': len(df[df['priority'] == 'high']),
            'understocked_percentage': len(df[df['inventory_status'] == 'Understocked']) / len(df) * 100
        }
        return trends
```

## Performance Optimization

### Caching Strategies

```python
from functools import lru_cache
import hashlib

class CachedShelfAnalyzer(ShelfAnalyzer):
    @lru_cache(maxsize=100)
    def _cached_analysis(self, image_hash, store_section):
        # Cache results based on image content hash
        return super().analyze_shelf(image_hash, store_section)
    
    def analyze_shelf(self, image_path, store_section="grocery"):
        # Generate content hash for caching
        with open(image_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()
        
        return self._cached_analysis(image_hash, store_section)
```

### Batch Processing

```python
def batch_analyze_shelves(image_paths, provider='claude'):
    analyzer = ShelfAnalyzer(provider=provider)
    results = {}
    
    # Process multiple shelves efficiently
    for shelf_id, image_path in image_paths.items():
        try:
            results[shelf_id] = analyzer.quick_analysis(image_path)
        except Exception as e:
            results[shelf_id] = f"Analysis failed: {e}"
            
    return results
```

## Troubleshooting

### Common Issues

**1. API Rate Limits**
```python
import time
import random

def analyze_with_retry(analyzer, image_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            return analyzer.analyze_shelf(image_path)
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            raise e
    raise Exception("Max retries exceeded")
```

**2. Poor Analysis Quality**
- Use more specific store section types
- Ensure high-quality, well-lit images
- Try different LLM providers for comparison
- Adjust provider temperature settings

**3. Parsing Failures**
- Implement custom parsing for specific business terms
- Add fallback logic for unparseable responses
- Use structured output formats when available

### Best Practices

1. **Image Quality**: Use well-lit, high-resolution images
2. **Section Specificity**: Provide accurate store section context
3. **Provider Selection**: Test different providers for your use case
4. **Error Handling**: Always implement graceful fallbacks
5. **Result Validation**: Cross-check AI insights with business logic
6. **Performance Monitoring**: Track analysis accuracy and response times