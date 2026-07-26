# Substack Analyzer 📊

A comprehensive Python data science project that analyzes Substack newsletters to:
- 🔮 **Predict future article topics** using NLP and machine learning
- 📚 **Recommend relevant books** based on content themes
- 📈 **Visualize growth areas** with interactive timelines
- 🎯 **Score your mastery** across consistency, depth, diversity, and growth
- 📊 **Interactive dashboard** with monthly filtering and real-time analytics

## Features

### 1. Article Prediction
- Uses TF-IDF vectorization and topic modeling (LDA & NMF)
- Identifies trending topics and content patterns
- Generates predictions for future article topics with confidence scores
- Analyzes topic evolution over time

### 2. Mastery Scoring (NEW! 🎯)
- **Consistency Score**: Publishing regularity and frequency
- **Depth Score**: Article length and content quality
- **Diversity Score**: Topic variety and vocabulary richness
- **Growth Score**: Improvement trajectory over time
- **Engagement Score**: Title effectiveness and content structure
- Overall mastery level from Beginner to Master
- Strengths, growth areas, and actionable recommendations

### 3. Book Recommendations
- Curated database of 25+ technical and educational books
- Topic-based relevance scoring
- Personalized recommendations based on historical content and predictions
- Explains why each book is recommended
- Filter by topics and relevance

### 4. Growth Visualization
- Interactive timeline showing content evolution
- Topic distribution charts
- Word clouds of key themes
- Trend analysis and prediction zones
- Monthly filtering for focused analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (first run only):
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### 🌟 Interactive Dashboard (Recommended)

Launch the interactive web dashboard for the best experience:

```bash
# Mac/Linux
./run_dashboard.sh

# Windows
run_dashboard.bat

# Or manually
streamlit run dashboard.py
```

The dashboard provides:
- Real-time analytics with monthly filtering
- Mastery scoring across 5 dimensions
- Topic suggestions timeline
- Personalized book recommendations
- Growth tracking visualizations

See [DASHBOARD_README.md](DASHBOARD_README.md) for complete dashboard documentation.

### Command Line Usage

Analyze a Substack newsletter:

```bash
python substack_analyzer.py --substack-url https://example.substack.com
```

### Advanced Options

```bash
python substack_analyzer.py \
    --substack-url https://example.substack.com \
    --output-dir ./my_analysis \
    --max-articles 100 \
    --num-predictions 10 \
    --num-books 15
```

### Parameters

- `--substack-url` (required): URL of the Substack newsletter
- `--output-dir` (optional): Directory for output files (default: `./output`)
- `--max-articles` (optional): Maximum number of articles to analyze (default: 50)
- `--num-predictions` (optional): Number of article predictions to generate (default: 5)
- `--num-books` (optional): Number of book recommendations (default: 10)

## Output Files

The analyzer generates the following files in the output directory:

1. **articles_data.csv** - Raw article data collected from Substack
2. **predictions.json** - Predicted future article topics with metadata
3. **book_recommendations.json** - Personalized book recommendations
4. **growth_timeline.html** - Interactive timeline visualization
5. **topic_evolution.html** - Topic trends over time
6. **wordcloud.png** - Visual word cloud of key themes
7. **summary_report.html** - Comprehensive HTML report
8. **analyzer.log** - Detailed execution logs

## Project Structure

```
Substack_Analyzer/
│
├── dashboard.py               # Interactive Streamlit dashboard (NEW!)
├── mastery_scorer.py          # Mastery scoring engine (NEW!)
├── substack_analyzer.py       # Command-line application
├── data_collector.py          # Substack data collection module
├── article_predictor.py       # ML-based article prediction
├── book_recommender.py        # Book recommendation engine
├── growth_visualizer.py       # Visualization generation
├── utils.py                   # Utility functions
├── example_usage.py           # Programmatic usage examples
├── test_analyzer.py           # Test suite
├── run_dashboard.sh           # Dashboard launcher (Mac/Linux)
├── run_dashboard.bat          # Dashboard launcher (Windows)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── DASHBOARD_README.md        # Dashboard documentation
├── QUICKSTART.md             # Quick start guide
└── config.example.json        # Configuration template
```

## How It Works

### 1. Data Collection
- Fetches articles via RSS feed or web scraping
- Extracts titles, content, dates, and tags
- Handles rate limiting and errors gracefully

### 2. Topic Modeling
- Applies TF-IDF vectorization to extract features
- Uses Latent Dirichlet Allocation (LDA) for topic discovery
- Employs Non-negative Matrix Factorization (NMF) for alternative topics
- Identifies topic trends over time

### 3. Prediction Algorithm
- Analyzes topic frequency and trends
- Identifies growing vs. declining topics
- Generates future article titles based on trending topics
- Calculates confidence scores for predictions

### 4. Book Recommendation
- Matches article topics with curated book database
- Scores books based on topic relevance
- Weights recent and predicted topics more heavily
- Provides explanations for each recommendation

### 5. Visualization
- Creates interactive Plotly visualizations
- Generates timeline with historical and predicted content
- Shows topic evolution and distribution
- Creates word clouds for quick theme overview

## Example Output

### Predictions
```json
{
  "title": "Exploring Neural Networks in Deep Learning",
  "confidence": 0.85,
  "topics": ["neural networks", "deep learning", "ai", "training", "optimization"],
  "topic_label": "Neural Networks & Deep Learning & AI",
  "description": "This article will likely explore themes around neural networks...",
  "estimated_interest": "High"
}
```

### Book Recommendations
```json
{
  "title": "Deep Learning",
  "author": "Ian Goodfellow, Yoshua Bengio, Aaron Courville",
  "topics": ["AI", "Machine Learning", "Deep Learning"],
  "relevance_score": 0.92,
  "reason": "Recommended based on your interest in AI, Machine Learning, Deep Learning"
}
```

## Technical Details

### Machine Learning Techniques
- **TF-IDF**: Term frequency-inverse document frequency for feature extraction
- **LDA**: Latent Dirichlet Allocation for probabilistic topic modeling
- **NMF**: Non-negative Matrix Factorization for parts-based topic discovery
- **Trend Analysis**: Time-series analysis of topic frequency

### Libraries Used
- **scikit-learn**: ML algorithms and text processing
- **pandas & numpy**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **matplotlib**: Static plots and word clouds
- **BeautifulSoup**: Web scraping
- **feedparser**: RSS feed parsing

## Limitations & Notes

1. **Data Access**: Some Substacks may have paywalls or restricted content
2. **Sample Data**: If no articles can be fetched, sample data is used for demonstration
3. **Prediction Quality**: Prediction accuracy depends on the amount and consistency of historical data
4. **API Rate Limits**: The tool respects rate limits and includes delays between requests

## Future Enhancements

- [ ] Integration with Substack API (when available)
- [ ] Support for multiple newsletters comparison
- [ ] Advanced NLP models (BERT, GPT-based)
- [ ] Sentiment analysis of articles
- [ ] Reader engagement prediction
- [ ] Export to various formats (PDF, Markdown)
- [ ] Web dashboard interface

## Troubleshooting

### No articles found
- Check that the Substack URL is correct
- Ensure the Substack has public RSS feed enabled
- Try with `--max-articles 10` to fetch fewer articles

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (3.8+ required)

### Visualization not opening
- Make sure you have a web browser installed
- Open HTML files manually from the output directory
- Check file permissions in the output directory

## Contributing

This is an educational data science project. Feel free to:
- Fork and modify for your needs
- Add new features or visualizations
- Improve the prediction algorithms
- Expand the book database

## License

This project is provided as-is for educational purposes.

## Contact

For questions or suggestions about this project, please open an issue or submit a pull request.

---

**Happy Analyzing! 🚀📊**
