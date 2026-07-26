"""
Substack Analyzer - Main Application
Analyzes Substack newsletters to predict future articles, recommend books, and visualize growth areas
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from data_collector import SubstackCollector
from article_predictor import ArticlePredictor
from book_recommender import BookRecommender
from growth_visualizer import GrowthVisualizer
from utils import setup_logging, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Substack newsletters and predict future content"
    )
    parser.add_argument(
        "--substack-url",
        type=str,
        required=True,
        help="Substack newsletter URL (e.g., https://example.substack.com)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="Maximum number of articles to analyze"
    )
    parser.add_argument(
        "--num-predictions",
        type=int,
        default=5,
        help="Number of article predictions to generate"
    )
    parser.add_argument(
        "--num-books",
        type=int,
        default=10,
        help="Number of book recommendations to generate"
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info(f"Starting Substack analysis for: {args.substack_url}")
    
    try:
        # Step 1: Collect data from Substack
        logger.info("Step 1: Collecting articles from Substack...")
        collector = SubstackCollector(args.substack_url)
        articles_df = collector.collect_articles(max_articles=args.max_articles)
        
        if articles_df.empty:
            logger.error("No articles found. Please check the Substack URL.")
            return 1
        
        logger.info(f"Collected {len(articles_df)} articles")
        articles_df.to_csv(output_dir / "articles_data.csv", index=False)
        
        # Step 2: Predict next articles
        logger.info("Step 2: Predicting future article topics...")
        predictor = ArticlePredictor()
        predictor.fit(articles_df)
        predictions = predictor.predict_next_articles(num_predictions=args.num_predictions)
        
        # Save predictions
        predictor.save_predictions(predictions, output_dir / "predictions.json")
        logger.info(f"Generated {len(predictions)} article predictions")
        
        # Step 3: Generate book recommendations
        logger.info("Step 3: Generating book recommendations...")
        recommender = BookRecommender()
        book_recommendations = recommender.recommend_books(
            articles_df,
            predictions,
            num_books=args.num_books
        )
        
        # Save recommendations
        recommender.save_recommendations(
            book_recommendations,
            output_dir / "book_recommendations.json"
        )
        logger.info(f"Generated {len(book_recommendations)} book recommendations")
        
        # Step 4: Visualize growth areas
        logger.info("Step 4: Creating growth timeline and visualizations...")
        visualizer = GrowthVisualizer()
        visualizer.create_growth_timeline(
            articles_df,
            predictions,
            output_dir / "growth_timeline.html"
        )
        visualizer.create_topic_evolution(
            articles_df,
            output_dir / "topic_evolution.html"
        )
        visualizer.create_wordcloud(
            articles_df,
            output_dir / "wordcloud.png"
        )
        
        # Generate summary report
        logger.info("Generating summary report...")
        generate_summary_report(
            articles_df,
            predictions,
            book_recommendations,
            output_dir / "summary_report.html"
        )
        
        logger.info(f"\n{'='*60}")
        logger.info("Analysis complete! Results saved to:")
        logger.info(f"  - Articles data: {output_dir / 'articles_data.csv'}")
        logger.info(f"  - Predictions: {output_dir / 'predictions.json'}")
        logger.info(f"  - Book recommendations: {output_dir / 'book_recommendations.json'}")
        logger.info(f"  - Growth timeline: {output_dir / 'growth_timeline.html'}")
        logger.info(f"  - Topic evolution: {output_dir / 'topic_evolution.html'}")
        logger.info(f"  - Word cloud: {output_dir / 'wordcloud.png'}")
        logger.info(f"  - Summary report: {output_dir / 'summary_report.html'}")
        logger.info(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return 1


def generate_summary_report(articles_df, predictions, book_recommendations, output_path):
    """Generate an HTML summary report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Substack Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .section {{
                background: white;
                padding: 30px;
                margin-bottom: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                margin-top: 0;
            }}
            .stat {{
                display: inline-block;
                margin: 10px 20px 10px 0;
                padding: 15px 25px;
                background: #f8f9fa;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }}
            .prediction {{
                padding: 15px;
                margin: 10px 0;
                background: #f8f9fa;
                border-left: 4px solid #28a745;
                border-radius: 5px;
            }}
            .book {{
                padding: 15px;
                margin: 10px 0;
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                border-radius: 5px;
            }}
            .timestamp {{
                color: #6c757d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Substack Analysis Report</h1>
            <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📈 Key Statistics</h2>
            <div class="stat">
                <strong>{len(articles_df)}</strong><br>
                Total Articles Analyzed
            </div>
            <div class="stat">
                <strong>{len(predictions)}</strong><br>
                Future Article Predictions
            </div>
            <div class="stat">
                <strong>{len(book_recommendations)}</strong><br>
                Book Recommendations
            </div>
        </div>
        
        <div class="section">
            <h2>🔮 Predicted Future Articles</h2>
            {''.join([f'''
            <div class="prediction">
                <strong>{i+1}. {pred['title']}</strong><br>
                <em>Confidence: {pred['confidence']:.1%}</em><br>
                Topics: {', '.join(pred['topics'][:5])}
            </div>
            ''' for i, pred in enumerate(predictions)])}
        </div>
        
        <div class="section">
            <h2>📚 Recommended Books</h2>
            {''.join([f'''
            <div class="book">
                <strong>{book['title']}</strong> by {book['author']}<br>
                <em>Relevance: {book['relevance_score']:.1%}</em><br>
                Reason: {book['reason']}
            </div>
            ''' for book in book_recommendations[:10]])}
        </div>
        
        <div class="section">
            <h2>📊 Visualizations</h2>
            <p>Open the following files for detailed visualizations:</p>
            <ul>
                <li><strong>growth_timeline.html</strong> - Interactive timeline of content growth areas</li>
                <li><strong>topic_evolution.html</strong> - Topic trends over time</li>
                <li><strong>wordcloud.png</strong> - Visual representation of key themes</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    sys.exit(main())
