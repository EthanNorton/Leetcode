# 🤖 LSTM Predictor with Time-Decay Weighting

## Overview

The LSTM (Long Short-Term Memory) predictor uses deep learning to predict future article topics with **time-decay weighting** that emphasizes recent content more heavily.

## Key Features

### 1. Time-Decay Weighting 📉
- Recent articles weighted exponentially more than older ones
- Decay rate configurable (0.85-0.99)
- Default: 0.95 (95% weighting per month)

**Example**: With decay_rate=0.95:
- Article from 1 month ago: 95% weight
- Article from 2 months ago: 90% weight  
- Article from 6 months ago: 77% weight
- Article from 1 year ago: 54% weight

### 2. LSTM Neural Network 🧠
- 2-layer LSTM with attention mechanism
- Attention focuses on recent time steps
- Sequence length: 5 articles (configurable)
- Hidden size: 64 neurons per layer

### 3. Hybrid Predictions 🔮
- Combines LSTM output with topic modeling
- Falls back to statistical methods if LSTM training fails
- Confidence scores based on attention weights

## Installation

```bash
pip install torch  # PyTorch for LSTM
pip install jupyter  # For interactive notebook
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Interactive Jupyter Notebook (Recommended)

```bash
./run_notebook.sh  # Mac/Linux
run_notebook.bat   # Windows

# Or manually
jupyter notebook Interactive_Substack_Analysis.ipynb
```

The notebook provides:
- Step-by-step interactive analysis
- LSTM training visualization
- Real-time predictions
- Experiment with different decay rates
- Full Python coding environment

### Option 2: Python Script

```python
from data_collector import SubstackCollector
from lstm_predictor import LSTMArticlePredictor

# Collect articles
collector = SubstackCollector("https://substack.com/@ethannorton")
articles_df = collector.collect_articles(max_articles=50)

# Initialize LSTM predictor
predictor = LSTMArticlePredictor(
    sequence_length=5,
    decay_rate=0.95  # Higher = more emphasis on recent
)

# Train and predict
predictor.fit(articles_df, train_model=True)
predictions = predictor.predict_next_articles(num_predictions=10)

# View results
for pred in predictions:
    print(f"{pred['title']} - {pred['confidence']:.0%}")
```

### Option 3: Integrate with Dashboard

The dashboard automatically uses LSTM if available:

```bash
streamlit run dashboard.py
```

## Configuration

### Decay Rate

Controls how much recent articles are weighted:

```python
decay_rate = 0.85  # Moderate emphasis on recent (15% decay/month)
decay_rate = 0.90  # Balanced (10% decay/month)
decay_rate = 0.95  # Strong emphasis (5% decay/month) - DEFAULT
decay_rate = 0.99  # Very strong emphasis (1% decay/month)
```

**Recommendation**: Use 0.95 for most cases, 0.99 if your content changes rapidly

### Sequence Length

Number of past articles to consider:

```python
sequence_length = 3   # Short memory (fast training)
sequence_length = 5   # Medium memory (balanced) - DEFAULT
sequence_length = 10  # Long memory (requires more data)
```

**Requirement**: You need at least `sequence_length + 1` articles to train

### LSTM Architecture

Customize the neural network:

```python
ArticleLSTM(
    input_size=200,      # TF-IDF features
    hidden_size=64,      # Neurons per layer (32-128)
    num_layers=2,        # LSTM layers (1-3)
    output_size=50       # Prediction features
)
```

## How It Works

### 1. Data Preparation
```
Articles → TF-IDF Features → Time-Weighted Features
```

For each article:
- Extract text (title + content + summary)
- Convert to TF-IDF vectors (200 features)
- Apply time-decay weights based on publish date

### 2. Sequence Creation
```
[Article 1, Article 2, Article 3, Article 4, Article 5] → Predict Article 6
```

Create sliding windows of articles for LSTM input.

### 3. LSTM Training
```
Input Sequence → LSTM Layers → Attention → Output Prediction
```

- Feed sequences through 2-layer LSTM
- Apply attention to focus on recent articles
- Train with weighted MSE loss (recent predictions weighted higher)
- 50 epochs with Adam optimizer

### 4. Prediction
```
Last 5 Articles → LSTM → Predicted Features → Article Topics
```

- Take most recent 5 articles as input
- Generate feature prediction
- Extract top keywords
- Generate article titles and descriptions

## Jupyter Notebook Features

The interactive notebook includes:

### 📊 Section 1: Configuration
- Set your Substack URL
- Configure decay rate
- Set max articles to analyze

### 📥 Section 2: Data Collection
- Fetch articles from your Substack
- Display article metadata
- Show date ranges

### 🎯 Section 3: Mastery Scoring
- Calculate your overall score
- View component breakdown
- Interactive gauge charts

### 🤖 Section 4: LSTM Predictions
- Train LSTM with progress display
- Generate predictions
- View attention weights
- Compare with statistical methods

### 📚 Section 5: Book Recommendations
- Get personalized reading list
- Filter by relevance

### 📈 Section 6: Analytics
- Publishing frequency charts
- Article length distribution
- Growth trends

### 🎯 Section 7: Content Roadmap
- Interactive timeline
- Past + future articles
- Visual roadmap

### 🔬 Section 8: Experiments
- Try different decay rates
- Compare results
- Tune parameters

## Output Format

Each prediction includes:

```python
{
    'title': 'The Future of AI: Understanding Perspective',
    'confidence': 0.87,  # 0-1 confidence score
    'topics': ['ai', 'understanding', 'systems', ...],
    'topic_label': 'AI & Understanding & Systems',
    'description': 'This article will explore...',
    'estimated_interest': 'High',
    'method': 'LSTM',  # or 'Statistical (Time-Weighted)'
    'attention_score': 0.92  # How much attention on recent
}
```

## Performance

### Training Time
- **5 articles**: < 1 second
- **50 articles**: 10-30 seconds
- **100 articles**: 30-60 seconds

*Depends on CPU/GPU and sequence settings*

### Accuracy
- Confidence scores: 50-95%
- Higher confidence = better alignment with your style
- LSTM typically 10-20% more confident than statistical

### Memory Usage
- Minimal: < 100MB for most cases
- Scales with number of articles and features

## Troubleshooting

### PyTorch Not Available
```
Warning: PyTorch not available. Install with: pip install torch
```

**Solution**: Install PyTorch
```bash
pip install torch
```

Falls back to statistical predictions automatically.

### Not Enough Articles
```
Not enough articles for LSTM training (need at least 6)
```

**Solution**: 
- Reduce `sequence_length` parameter
- Or collect more articles with `max_articles`

### Jupyter Not Opening
```
jupyter: command not found
```

**Solution**:
```bash
pip install jupyter notebook
```

### Slow Training
**Solutions**:
- Reduce `num_epochs` (default: 50)
- Reduce `hidden_size` (default: 64)
- Reduce `max_articles` to analyze fewer articles

## Comparison: LSTM vs Statistical

| Feature | LSTM | Statistical |
|---------|------|-------------|
| **Speed** | Slower (training) | Fast |
| **Accuracy** | Higher | Good |
| **Data Required** | 6+ articles | 3+ articles |
| **Temporal Awareness** | Strong | Moderate |
| **Confidence** | 70-95% | 50-80% |
| **Dependencies** | PyTorch | Scikit-learn only |

## Best Practices

1. **Data Quality**: More articles = better predictions
2. **Decay Rate**: Higher for fast-changing content (0.95-0.99)
3. **Training**: Let LSTM train (improves over statistical)
4. **Experimentation**: Use notebook to test different settings
5. **Interpretation**: High confidence = strong pattern match

## Integration with Existing Tools

### With Dashboard
```python
# In dashboard.py
from lstm_predictor import LSTMArticlePredictor

# Use LSTM instead of standard predictor
predictor = LSTMArticlePredictor(decay_rate=0.95)
predictor.fit(articles_df)
predictions = predictor.predict_next_articles(10)
```

### With Command Line
```bash
python substack_analyzer.py --use-lstm --decay-rate 0.95
```

### With Mastery Scorer
```python
# Combine LSTM predictions with mastery analysis
mastery_results = scorer.analyze_mastery(articles_df, articles_df)
lstm_predictions = predictor.predict_next_articles(10)

# Use both for comprehensive insights
```

## Examples

### Example 1: Quick Analysis
```python
predictor = LSTMArticlePredictor()
predictor.fit(articles_df)
predictions = predictor.predict_next_articles(5)
```

### Example 2: Emphasize Recent Content
```python
predictor = LSTMArticlePredictor(decay_rate=0.99)
predictor.fit(articles_df)
predictions = predictor.predict_next_articles(10)
```

### Example 3: Compare Decay Rates
```python
for decay in [0.85, 0.90, 0.95, 0.99]:
    predictor = LSTMArticlePredictor(decay_rate=decay)
    predictor.fit(articles_df, train_model=False)
    preds = predictor.predict_next_articles(3)
    print(f"Decay {decay}: {preds[0]['title']}")
```

## Future Enhancements

- [ ] Transformer-based models (BERT, GPT)
- [ ] Multi-task learning (predict engagement too)
- [ ] Transfer learning from similar Substacks
- [ ] Automatic hyperparameter tuning
- [ ] GPU acceleration support
- [ ] Model persistence (save/load trained models)

---

**Ready to use LSTM predictions? Open the Jupyter notebook and start experimenting!** 🚀
