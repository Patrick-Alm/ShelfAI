# ShelfAI 🛒

**AI-powered grocery shelf analysis using computer vision and large language models**

ShelfAI combines YOLOv8 object detection with LLM analysis to provide comprehensive retail shelf insights for inventory management, product placement optimization, and customer experience improvement.

## 🌟 Features

- **Advanced Product Detection**: YOLOv8-based detection with enhanced preprocessing
- **Multi-Provider LLM Analysis**: Support for Claude, Gemini, and OpenAI
- **Comprehensive Reporting**: JSON reports with detailed metrics and recommendations
- **Flexible Usage**: CLI tool and Python package for integration
- **Visualization**: Detection bounding boxes with confidence scores
- **Real-time Analysis**: Fast processing with GPU acceleration (MPS/CUDA)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:Patrick-Alm/PlantAI.git
cd PlantAI

# Install with uv (recommended)
uv sync
uv sync --group dev  # Include development dependencies

# Or with pip
pip install -e .
```

### CLI Usage

```bash
# Basic analysis
shelfai shelf_image.jpg

# Use specific LLM provider
shelfai shelf_image.jpg --provider claude

# Save detailed report
shelfai shelf_image.jpg --save-report analysis.json

# Create visualization with lower confidence
shelfai shelf_image.jpg --confidence 0.3 --visualize

# Complete analysis with all options
shelfai shelf_image.jpg --provider gemini --confidence 0.25 --model-size x --visualize --save-report detailed_report.json
```

### Python Package Usage

```python
from PlantAI import ProductDetector, ShelfAnalyzer

# Initialize components
detector = ProductDetector(confidence=0.4, model_size='x')
analyzer = ShelfAnalyzer(provider='claude')

# Run analysis
detection_results = detector.detect_products('shelf_image.jpg')
ai_analysis = analyzer.analyze_shelf('shelf_image.jpg')

# Access results
print(f"Detected {detection_results['total_detections']} products")
print(f"Shelf density: {detection_results['shelf_density']}")
print(f"AI Recommendations: {ai_analysis}")
```

## 📊 Output Example

```json
{
  "metadata": {
    "timestamp": "2025-01-06T15:30:45.123456",
    "provider": "claude",
    "confidence_threshold": 0.4
  },
  "detection_results": {
    "total_detections": 25,
    "food_items": 15,
    "beverages": 8,
    "household_items": 2,
    "other_products": 0,
    "shelf_density": "high",
    "food_percentage": 60.0
  },
  "ai_analysis": "Comprehensive retail recommendations...",
  "summary": {
    "total_products": 25,
    "product_distribution": {...}
  }
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Required for LLM providers
export ANTHROPIC_API_KEY="your-claude-key"
export GOOGLE_API_KEY="your-gemini-key" 
export OPENAI_API_KEY="your-openai-key"
```

### Model Configuration

- **Detection Models**: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- **LLM Models**: 
  - Claude: `claude-3-5-sonnet-20241022`
  - Gemini: `gemini-1.5-flash`
  - OpenAI: `gpt-4o-mini`

## 📋 CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | LLM provider (claude/gemini/openai) | `openai` |
| `--confidence` | Detection confidence threshold | `0.4` |
| `--model-size` | YOLO model size (n/s/m/l/x) | `x` |
| `--visualize` | Save detection visualization | `False` |
| `--output` | Visualization output path | Auto-generated |
| `--save-report` | Save JSON analysis report | None |

## 🏗️ Architecture

### Core Components

1. **ProductDetector** (`detector.py`): YOLO-based product detection
2. **ShelfAnalyzer** (`analyzer.py`): LLM integration for retail insights  
3. **Providers** (`providers/`): Multi-LLM client implementations
4. **Main** (`main.py`): CLI entry point

### Detection Flow

```
Image → Preprocessing → YOLOv8 Detection → Classification → Analysis → LLM Insights
```

### Key Features

- **Image Preprocessing**: CLAHE enhancement, sharpening, noise reduction
- **Multi-scale Detection**: High-resolution processing for small objects
- **Product Classification**: Food, beverages, household items, other
- **Shelf Density Analysis**: Very low to very high density levels
- **Device Optimization**: Automatic MPS/CUDA/CPU selection

## 📚 Documentation

- [ProductDetector Documentation](docs/detector.md)
- [ShelfAnalyzer Documentation](docs/analyzer.md)
- [Provider System Documentation](docs/providers.md)

## 🔬 Development

### Running Tests

```bash
uv run pytest
```

### Development Setup

```bash
uv sync --group dev
```

## 📈 Performance Tips

1. **Use YOLOv8x** for maximum accuracy on complex shelves
2. **Lower confidence threshold** (0.2-0.3) for dense product displays
3. **Enable visualization** to verify detection quality
4. **Use MPS/CUDA** for faster processing on supported hardware

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes as part of a college assignment.

## 🔗 Links

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Google Gemini API](https://ai.google.dev/)
- [OpenAI API](https://platform.openai.com/docs/)
