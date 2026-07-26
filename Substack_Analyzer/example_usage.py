"""
Example usage script for Substack Analyzer
Demonstrates how to use the analyzer programmatically
"""

from pathlib import Path
from data_collector import SubstackCollector
from article_predictor import ArticlePredictor
from book_recommender import BookRecommender
from growth_visualizer import GrowthVisualizer

def analyze_substack(substack_url: str):
    """
    Example function showing programmatic usage
    
    Args:
        substack_url: URL of the Substack to analyze
    """
    print(f"Analyzing Substack: {substack_url}\n")
    
    # Setup output directory
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Collect articles
    print("="*60)
    print("STEP 1: Collecting Articles")
    print("="*60)
    collector = SubstackCollector(substack_url)
    articles_df = collector.collect_articles(max_articles=30)
    print(f"✓ Collected {len(articles_df)} articles\n")
    
    # Display sample articles
    print("Sample Articles:")
    for idx, row in articles_df.head(3).iterrows():
        print(f"  {idx+1}. {row['title']}")
        print(f"     Published: {row['published_date']}")
        print()
    
    # Step 2: Predict future articles
    print("="*60)
    print("STEP 2: Predicting Future Articles")
    print("="*60)
    predictor = ArticlePredictor()
    predictor.fit(articles_df)
    predictions = predictor.predict_next_articles(num_predictions=5)
    
    print("Predicted Future Articles:")
    for i, pred in enumerate(predictions, 1):
        print(f"\n  {i}. {pred['title']}")
        print(f"     Confidence: {pred['confidence']:.1%}")
        print(f"     Topics: {', '.join(pred['topics'][:5])}")
    print()
    
    # Step 3: Get book recommendations
    print("="*60)
    print("STEP 3: Generating Book Recommendations")
    print("="*60)
    recommender = BookRecommender()
    books = recommender.recommend_books(articles_df, predictions, num_books=5)
    
    print("Top Book Recommendations:")
    for i, book in enumerate(books, 1):
        print(f"\n  {i}. {book['title']}")
        print(f"     Author: {book['author']}")
        print(f"     Relevance: {book['relevance_score']:.1%}")
        print(f"     Reason: {book['reason']}")
    print()
    
    # Step 4: Create visualizations
    print("="*60)
    print("STEP 4: Creating Visualizations")
    print("="*60)
    visualizer = GrowthVisualizer()
    
    print("Creating growth timeline...")
    visualizer.create_growth_timeline(
        articles_df,
        predictions,
        output_dir / "growth_timeline.html"
    )
    
    print("Creating topic evolution chart...")
    visualizer.create_topic_evolution(
        articles_df,
        output_dir / "topic_evolution.html"
    )
    
    print("Creating word cloud...")
    visualizer.create_wordcloud(
        articles_df,
        output_dir / "wordcloud.png"
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir.absolute()}")
    print("\nOpen the following files to view results:")
    print(f"  - {output_dir / 'growth_timeline.html'}")
    print(f"  - {output_dir / 'topic_evolution.html'}")
    print(f"  - {output_dir / 'wordcloud.png'}")
    print()


if __name__ == "__main__":
    # Example Substack URLs (replace with actual Substack)
    example_urls = [
        "https://stratechery.com",  # Tech analysis
        "https://astralcodexten.substack.com",  # AI and rationality
        "https://every.to",  # Business and productivity
    ]
    
    # Use the first example or provide your own
    substack_url = input("Enter Substack URL (or press Enter for demo): ").strip()
    
    if not substack_url:
        print("Using demo with sample data...\n")
        substack_url = "https://example.substack.com"
    
    analyze_substack(substack_url)
