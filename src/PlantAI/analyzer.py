from typing import Dict, Optional
from .detector import ProductDetector
from .providers.claude_provider import ClaudeClient
from .providers.gemini_provider import GeminiClient
from .providers.openai_provider import OpenAIClient


class ShelfAnalyzer:
    def __init__(self, provider: str = "claude", api_key: Optional[str] = None, **kwargs):
        self.detector = ProductDetector()
        
        self.provider_name = provider.lower()
        
        if self.provider_name == "claude":
            self.client = ClaudeClient(api_key=api_key, **kwargs)
        elif self.provider_name == "gemini":
            self.client = GeminiClient(api_key=api_key, **kwargs)
        elif self.provider_name == "openai":
            self.client = OpenAIClient(api_key=api_key, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'claude', 'gemini', or 'openai'")

    def analyze_shelf(self, image_path: str, store_section: str = "grocery") -> Dict:
        detection_results = self.detector.detect_products(image_path)
        insights = self._get_retail_insights(detection_results, store_section)

        return {
            'detection_results': detection_results,
            'insights': insights,
            'summary': self._create_summary(detection_results, insights)
        }

    def _get_retail_insights(self, detection_results: Dict, store_section: str) -> Dict:
        prompt = f"""
Supermarket Shelf Analysis Report:

Store Section: {store_section}
Total products detected: {detection_results['total_detections']}
Food items: {detection_results['food_items']} 
Beverages: {detection_results['beverages']}
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

        try:
            advice = self.client.generate_response(prompt)
            parsed = self._parse_insights(advice)

            return {
                'full_insights': advice,
                'inventory_status': parsed.get('inventory', 'Review full insights'),
                'product_mix': parsed.get('mix', 'Assess product variety'),
                'restocking_priority': parsed.get('priority', 'medium'),
                'organization': parsed.get('organization', 'Review shelf layout'),
                'action_steps': parsed.get('steps', ['Review full insights'])
            }

        except Exception as e:
            return {
                'full_insights': f"Error: {e}",
                'inventory_status': 'Manual assessment needed',
                'product_mix': 'Consult store manager',
                'restocking_priority': 'unknown',
                'organization': 'Review manually',
                'action_steps': ['Consult store management']
            }

    def _parse_insights(self, insights: str) -> Dict:
        parsed = {}
        lines = insights.lower().split('\n')

        for line in lines:
            if 'well-stocked' in line or 'good stock' in line:
                parsed['inventory'] = 'Well-stocked'
            elif 'understocked' in line or 'low stock' in line:
                parsed['inventory'] = 'Understocked'
            elif 'overstocked' in line:
                parsed['inventory'] = 'Overstocked'
            elif 'high priority' in line or 'urgent' in line:
                parsed['priority'] = 'high'
            elif 'low priority' in line:
                parsed['priority'] = 'low'
            elif 'good variety' in line or 'diverse' in line:
                parsed['mix'] = 'Good product variety'
            elif 'poor variety' in line or 'limited' in line:
                parsed['mix'] = 'Limited variety'
            elif 'reorganize' in line or 'rearrange' in line:
                parsed['organization'] = 'Needs reorganization'

        return parsed

    def _create_summary(self, detection_results: Dict, insights: Dict) -> str:
        return f"""
🛒 SHELF ANALYSIS SUMMARY
========================

Detection Results:
• {detection_results['total_detections']} products detected
• {detection_results['food_percentage']}% food items
• Shelf density: {detection_results['shelf_density'].upper().replace('_', ' ')}

Retail Insights:
• Inventory: {insights['inventory_status']}
• Product mix: {insights['product_mix']}
• Priority: {insights['restocking_priority'].upper()}
• Organization: {insights['organization']}

Next Steps:
{chr(10).join('• ' + step for step in insights['action_steps'])}
"""

    def quick_analysis(self, image_path: str, store_section: str = "grocery") -> str:
        try:
            result = self.analyze_shelf(image_path, store_section)
            return result['summary']
        except Exception as e:
            return f"Analysis failed: {e}"

    def save_results(self, analysis_result: Dict, output_file: str = "shelf_analysis.txt"):
        with open(output_file, 'w') as f:
            f.write("RETAIL SHELF ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(analysis_result['summary'])
            f.write("\n\nFULL INSIGHTS:\n")
            f.write("-" * 30 + "\n")
            f.write(analysis_result['insights']['full_insights'])

        print(f"Results saved to: {output_file}")