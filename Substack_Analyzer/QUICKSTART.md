# Quick Start Guide 🚀

Get started with Substack Analyzer in 5 minutes!

## Installation

```bash
# 1. Navigate to the project directory
cd Substack_Analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (one-time setup)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage Examples

### Example 1: Basic Analysis

Analyze any Substack newsletter:

```bash
python substack_analyzer.py --substack-url https://example.substack.com
```

### Example 2: Quick Demo

Run the interactive example:

```bash
python example_usage.py
```

Press Enter when prompted to use sample data for a quick demo.

### Example 3: Detailed Analysis

Analyze with custom parameters:

```bash
python substack_analyzer.py \
    --substack-url https://your-favorite-newsletter.substack.com \
    --max-articles 100 \
    --num-predictions 10 \
    --num-books 15 \
    --output-dir ./my_analysis
```

## What You'll Get

After running the analyzer, you'll find these files in the `output/` directory:

1. **summary_report.html** - Open this first! Comprehensive overview of results
2. **growth_timeline.html** - Interactive timeline showing content evolution
3. **topic_evolution.html** - How topics have changed over time
4. **wordcloud.png** - Visual representation of key themes
5. **predictions.json** - Detailed predictions data
6. **book_recommendations.json** - Full book recommendation list
7. **articles_data.csv** - Raw article data

## Viewing Results

### On Desktop
Simply double-click the HTML files to open them in your browser.

### On Terminal
```bash
# Open summary report
open output/summary_report.html  # macOS
xdg-open output/summary_report.html  # Linux
start output/summary_report.html  # Windows
```

## Real-World Examples

### Analyze a Tech Newsletter
```bash
python substack_analyzer.py --substack-url https://stratechery.com
```

### Analyze a Data Science Newsletter
```bash
python substack_analyzer.py --substack-url https://towardsdatascience.com
```

## Troubleshooting

### Problem: "No articles found"
**Solution:** The Substack might not have an accessible RSS feed. The analyzer will use sample data for demonstration.

### Problem: "Module not found"
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Problem: Visualizations don't open
**Solution:** Navigate to the `output/` directory and manually open the HTML files.

## Next Steps

1. ✅ Run the basic example
2. ✅ View the generated visualizations
3. ✅ Try with your favorite Substack
4. ✅ Explore the code to customize predictions
5. ✅ Modify the book database in `book_recommender.py`

## Need Help?

- Read the full `README.md` for detailed documentation
- Check the example code in `example_usage.py`
- Review the configuration in `config.example.json`

## Tips for Best Results

1. **More articles = Better predictions**: Use `--max-articles 100` for larger newsletters
2. **Consistent content**: Works best with newsletters that have regular themes
3. **Technical content**: Book recommendations are optimized for technical/educational content
4. **Time analysis**: Newsletters with dates allow for better trend analysis

---

**Ready to analyze? Pick a Substack and run the analyzer!** 🎯
