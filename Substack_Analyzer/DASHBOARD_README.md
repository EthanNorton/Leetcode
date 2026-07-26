# 📊 Interactive Dashboard Guide

## Overview

The Substack Analyzer Dashboard is a powerful interactive web application that provides:

- **📊 Real-time Analytics**: Track your publishing patterns and content metrics
- **🎯 Mastery Scoring**: Get scored on consistency, depth, diversity, growth, and engagement
- **💡 Topic Suggestions**: AI-powered predictions for your next articles
- **📚 Book Recommendations**: Personalized reading list based on your content
- **📅 Monthly Filtering**: Analyze your progress by specific time periods
- **📈 Growth Tracking**: Visualize your improvement over time

## Quick Start

### Option 1: Using Launch Scripts (Easiest)

**On Mac/Linux:**
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**On Windows:**
```bash
run_dashboard.bat
```

### Option 2: Manual Launch

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## Dashboard Features

### 1. 📊 Overview Tab

**Key Metrics:**
- Total articles published
- Average article length
- Publishing frequency
- Days/months active

**Visualizations:**
- Monthly publishing frequency chart
- Recent articles list with summaries
- Publishing trend analysis

### 2. 🎯 Mastery Score Tab

**Overall Mastery Score:**
- Circular gauge showing your current level (0-100)
- Mastery level badge (Beginner → Master)
- Score breakdown by component

**Five Scoring Dimensions:**

1. **Consistency Score (25%)**: Publishing regularity and frequency
2. **Depth Score (20%)**: Article length and content quality
3. **Diversity Score (20%)**: Topic variety and vocabulary richness
4. **Growth Score (20%)**: Improvement trajectory over time
5. **Engagement Score (15%)**: Title effectiveness and content structure

**Detailed Insights:**
- ✅ **Strengths**: Your top 3 areas of excellence
- 🎯 **Growth Areas**: 3 areas to focus on improving
- 📈 **Recommendations**: Actionable advice for improvement
- 🎓 **Topic Expertise**: Your expertise level in different topics

### 3. 💡 Topic Suggestions Tab

**Content Roadmap:**
- Interactive timeline showing past articles and future suggestions
- Predictions based on trending topics and content gaps
- Confidence scores for each suggestion

**Detailed Suggestions:**
- 📝 Suggested article titles
- 🎯 Key topics to cover
- 📊 Confidence and priority levels
- 💡 Detailed descriptions

**Features:**
- Copy suggested titles for easy use
- Filter by confidence level
- View suggested publishing dates

### 4. 📚 Book Recommendations Tab

**Personalized Reading List:**
- Books matched to your content themes
- Relevance scores (0-100%)
- Topic-based filtering
- Detailed explanations for each recommendation

**Filter Options:**
- By specific topics
- Minimum relevance threshold
- Topic combinations

**Book Details:**
- Title and author
- Relevant topics
- Relevance score visualization
- Reasoning for recommendation

### 5. 📈 Growth Analytics Tab

**Topic Evolution:**
- Track how topics have evolved over time
- Cumulative growth charts
- Identify trending themes

**Content Depth Analysis:**
- Distribution of article lengths
- Word count statistics
- Quality consistency metrics

**Writing Consistency:**
- Articles per month trend
- 3-month rolling average
- Consistency scoring

## Using Monthly Filters

The dashboard includes powerful date filtering:

1. **Sidebar Filter**: Select "Filter by Month"
2. **Options**: Choose "All Time" or any specific month
3. **Live Updates**: All tabs update automatically
4. **Comparisons**: See how each month compares

## Interpreting Mastery Scores

### Score Ranges & Levels

| Score | Level | What It Means |
|-------|-------|---------------|
| 0-20 | Beginner | Just starting out, building foundation |
| 20-40 | Developing | Establishing habits, finding voice |
| 40-60 | Intermediate | Consistent output, developing expertise |
| 60-75 | Advanced | Strong portfolio, recognized expertise |
| 75-85 | Expert | Thought leader, highly consistent quality |
| 85-100 | Master | Peak performance, mastery achieved |

### Component Scores Explained

**Consistency Score:**
- Measures how regularly you publish
- Factors: Publishing gaps, recent activity
- **Goal**: Minimize variance, publish regularly

**Depth Score:**
- Evaluates content thoroughness
- Factors: Article length, word count
- **Goal**: 1500+ words per article

**Diversity Score:**
- Assesses topic variety
- Factors: Unique vocabulary, topic range
- **Goal**: Broad expertise, varied content

**Growth Score:**
- Tracks improvement over time
- Factors: Increasing length, frequency
- **Goal**: Continuous improvement

**Engagement Score:**
- Estimates reader appeal
- Factors: Title effectiveness, structure
- **Goal**: Compelling, well-structured content

## Tips for Best Results

### 1. Data Quality
- More articles = better predictions
- Regular publishing = better trends
- Complete content = accurate scoring

### 2. Mastery Improvement
- **Consistency**: Set a publishing schedule
- **Depth**: Aim for 1500+ words
- **Diversity**: Explore related topics
- **Growth**: Track improvements monthly
- **Engagement**: Use power words in titles

### 3. Using Predictions
- Higher confidence = better aligned with your style
- Use suggested topics as inspiration
- Adapt predictions to your unique voice

### 4. Book Recommendations
- Start with highest relevance scores
- Filter by topics you want to grow in
- Use books to expand content themes

## Configuration

### Sidebar Settings

**Substack URL:**
- Enter your Substack profile or newsletter URL
- Default: `https://substack.com/@ethannorton`

**Maximum Articles:**
- Range: 10-200 articles
- More articles = better analysis
- Recommended: 50-100 for balance

**Topic Suggestions:**
- Range: 5-20 suggestions
- Customize based on your planning horizon
- Recommended: 10 suggestions

**Book Recommendations:**
- Range: 5-30 books
- More = broader reading list
- Recommended: 15 books

## Troubleshooting

### Dashboard Won't Start

**Problem**: Import errors
```bash
# Solution: Install dependencies
pip3 install -r requirements.txt
```

**Problem**: Port already in use
```bash
# Solution: Use different port
streamlit run dashboard.py --server.port 8502
```

### No Data Showing

**Problem**: Can't fetch Substack data
- Check your Substack URL is correct
- Ensure Substack has RSS feed enabled
- Dashboard will use sample data as fallback

### Slow Loading

**Problem**: Large dataset
- Reduce max_articles in sidebar
- Use monthly filters to focus analysis
- Clear Streamlit cache (press 'C' in dashboard)

### Visualizations Not Displaying

**Problem**: Browser compatibility
- Use Chrome, Firefox, or Safari
- Enable JavaScript
- Clear browser cache

## Keyboard Shortcuts

- **R**: Rerun the dashboard
- **C**: Clear cache
- **Ctrl+C**: Stop the dashboard (in terminal)
- **F5**: Refresh browser

## Exporting Data

### Export Options

**From Overview Tab:**
- Download article data as CSV
- Copy metrics for reports

**From Mastery Tab:**
- Screenshot overall score
- Copy recommendations

**From Suggestions Tab:**
- Copy individual titles
- Export predictions (coming soon)

**From Books Tab:**
- Add to reading list
- Export recommendations (coming soon)

## Privacy & Data

- **Local Processing**: All analysis runs on your computer
- **No Data Sent**: Nothing is shared externally
- **Cache**: Data cached locally for 1 hour
- **Security**: Your Substack data stays private

## Advanced Usage

### Custom Analysis Periods

Use the monthly filter to analyze:
- **Launch months**: Track initial traction
- **Peak months**: Identify what worked
- **Recent months**: Focus on current performance

### Tracking Progress

1. **Monthly Reviews**: Check mastery score each month
2. **Quarterly Goals**: Set targets for each component
3. **Annual Retrospective**: Review full year trends

### Growth Strategy

1. **Baseline**: Run analysis with all articles
2. **Identify**: Focus on lowest scoring areas
3. **Improve**: Implement recommendations
4. **Track**: Re-run monthly to measure progress

## Support & Feedback

For issues or suggestions:
1. Check this documentation
2. Review main README.md
3. Check example_usage.py for code examples

## Next Steps

1. ✅ Launch the dashboard
2. ✅ Enter your Substack URL
3. ✅ Explore each tab
4. ✅ Check your mastery score
5. ✅ Review topic suggestions
6. ✅ Add books to reading list
7. ✅ Set improvement goals

---

**Ready to level up your Substack? Launch the dashboard and start analyzing!** 🚀
