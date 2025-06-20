import argparse
import sys
import json
from datetime import datetime
from pathlib import Path
from .detector import ProductDetector
from .analyzer import ShelfAnalyzer


def main():
    """Main entry point for ShelfAI CLI"""
    parser = argparse.ArgumentParser(
        description="ShelfAI - Analyze grocery shelf products using computer vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  shelfai image.jpg                    # Basic analysis with OpenAI
  shelfai image.jpg --provider claude  # Use Claude for analysis
  shelfai image.jpg --confidence 0.3   # Lower confidence threshold
  shelfai image.jpg --visualize        # Save detection visualization
  shelfai image.jpg --save-report analysis.json  # Save full analysis report
        """
    )
    
    parser.add_argument(
        "image", 
        type=str,
        help="Path to shelf image to analyze"
    )
    
    parser.add_argument(
        "--provider",
        choices=["claude", "gemini", "openai"],
        default="openai",
        help="LLM provider for analysis (default: openai)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold (default: 0.4)"
    )
    
    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m", "l", "x"],
        default="x",
        help="YOLO model size (default: x for best accuracy)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization with bounding boxes"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for visualization (default: auto-generated)"
    )
    
    parser.add_argument(
        "--save-report",
        type=str,
        help="Save detailed analysis report to JSON file"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file '{args.image}' not found")
        sys.exit(1)

    try:
        detector = ProductDetector(
            confidence=args.confidence,
            model_size=args.model_size
        )

        detection_results = detector.detect_products(str(image_path))

        if args.visualize:
            if args.output:
                output_path = args.output
                if not output_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    output_path += '.jpg'
            else:
                output_path = str(image_path).replace('.jpg', '_detected.jpg')
            detector.visualize(str(image_path), output_path)

        analyzer = ShelfAnalyzer(provider=args.provider)
        analysis = analyzer.analyze_shelf(str(image_path))

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "image_path": str(image_path),
                "provider": args.provider,
                "model_size": args.model_size,
                "confidence_threshold": args.confidence
            },
            "detection_results": detection_results,
            "ai_analysis": analysis,
            "summary": {
                "total_products": detection_results['total_detections'],
                "product_distribution": {
                    "food_items": detection_results['food_items'],
                    "beverages": detection_results['beverages'],
                    "household_items": detection_results['household_items'],
                    "other_products": detection_results['other_products']
                },
                "shelf_density": detection_results['shelf_density'],
                "food_percentage": detection_results['food_percentage']
            }
        }

        if args.save_report:
            try:
                with open(args.save_report, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"Analysis report saved to: {args.save_report}")
            except Exception as e:
                print(f"Error saving report: {e}")

        print("\n" + "="*60)
        print("SHELF ANALYSIS RESULTS")
        print("="*60)
        print(f"Total products detected: {detection_results['total_detections']}")
        print(f"Food items: {detection_results['food_items']}")
        print(f"Beverages: {detection_results['beverages']}")
        print(f"Household items: {detection_results['household_items']}")
        print(f"Other products: {detection_results['other_products']}")
        print(f"Shelf density: {detection_results['shelf_density']}")
        print(f"Food percentage: {detection_results['food_percentage']}%")
        print("\n" + "-"*60)
        print("AI RECOMMENDATIONS:")
        print("-"*60)
        print(analysis)
        print("="*60)

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()