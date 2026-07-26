# 📰 Substack Content Dashboard

A powerful dashboard for fetching and analyzing Substack posts with **3-month period selection**.

## Features

✅ **Fetch Substack Content by Username** - Search and load posts from any Substack publication  
✅ **3-Month Period Selection** - View content in quarterly intervals (Q1, Q2, Q3, Q4)  
✅ **Multiple Period Selection** - Select multiple 3-month periods at once  
✅ **Interactive Dashboard** - Beautiful visualizations and analytics  
✅ **Content Analytics** - Word counts, post types, timelines, and distributions  

## Installation

1. Install dependencies:
```bash
pip install -r requirements_substack.txt
```

## Usage

### Run the Dashboard

```bash
streamlit run substack_dashboard.py
```

### Using the Interface

1. **Enter Substack Username** - Type the username in the sidebar (e.g., `ethanmorton`)
2. **Select Year Range** - Choose start and end years for period generation
3. **Select 3-Month Periods** - Choose one or more quarterly periods (e.g., "2024 Q3 (Jul-Sep)")
4. **Load Content** - Click "Load Substack Content" button
5. **Explore Results** - View timeline, post list, and analytics

### Programmatic Usage

```python
from substack_fetcher import SubstackFetcher
from datetime import datetime

# Initialize fetcher
fetcher = SubstackFetcher("ethanmorton")

# Get 3-month periods
periods = fetcher.get_three_month_periods(2024)

# Fetch posts for a specific period
start_date = datetime(2024, 7, 1)
end_date = datetime(2024, 9, 30)
posts = fetcher.fetch_posts_by_date_range(start_date, end_date)

# Display results
for post in posts:
    print(f"{post['title']} - {post['post_date']}")
```

## Dashboard Tabs

### 📈 Timeline
- Posts distribution by 3-month period
- Word count trends over quarters
- Interactive bar and line charts

### 📋 Post List
- Detailed post information
- Filter by period
- Direct links to read posts
- Metadata (date, word count, type)

### 📊 Analytics
- Post type distribution
- Audience distribution
- Word count statistics (min, max, mean, median)
- Word count histogram

## 3-Month Period Format

Periods are organized in quarters:
- **Q1 (Jan-Mar)** - January 1 to March 31
- **Q2 (Apr-Jun)** - April 1 to June 30
- **Q3 (Jul-Sep)** - July 1 to September 30
- **Q4 (Oct-Dec)** - October 1 to December 31

Example: `2024 Q3 (Jul-Sep)`

## API Details

The tool uses the Substack public API:
- Endpoint: `https://{username}.substack.com/api/v1/archive`
- Returns post metadata including titles, dates, word counts, and URLs
- No authentication required for public posts

## Troubleshooting

### Content Not Loading?
- Verify the username is correct (check the Substack URL)
- Ensure the publication has public posts
- Check internet connection

### No Posts Found?
- Try a different time period
- The publication may not have posts in the selected quarters

## Example Output

```
Found 15 posts
Total Words: 45,230
Average Words per Post: 3,015
```

## Tech Stack

- **Python 3.8+**
- **Streamlit** - Interactive dashboard
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **Requests** - HTTP requests to Substack API

## License

MIT License
