"""
Simple test script to verify the Substack Analyzer works correctly
"""

import sys
from pathlib import Path
import pandas as pd

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from data_collector import SubstackCollector
        from article_predictor import ArticlePredictor
        from book_recommender import BookRecommender
        from growth_visualizer import GrowthVisualizer
        from utils import setup_logging, load_config
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_collector():
    """Test data collection with sample data"""
    print("\nTesting data collector...")
    try:
        from data_collector import SubstackCollector
        
        collector = SubstackCollector("https://example.substack.com")
        articles_df = collector.collect_articles(max_articles=5)
        
        assert len(articles_df) > 0, "No articles collected"
        assert 'title' in articles_df.columns, "Missing 'title' column"
        assert 'content' in articles_df.columns, "Missing 'content' column"
        
        print(f"✓ Collected {len(articles_df)} sample articles")
        return True
    except Exception as e:
        print(f"✗ Data collector error: {e}")
        return False

def test_article_predictor():
    """Test article prediction"""
    print("\nTesting article predictor...")
    try:
        from data_collector import SubstackCollector
        from article_predictor import ArticlePredictor
        
        # Get sample data
        collector = SubstackCollector("https://example.substack.com")
        articles_df = collector.collect_articles(max_articles=5)
        
        # Test prediction
        predictor = ArticlePredictor()
        predictor.fit(articles_df)
        predictions = predictor.predict_next_articles(num_predictions=3)
        
        assert len(predictions) == 3, f"Expected 3 predictions, got {len(predictions)}"
        assert 'title' in predictions[0], "Missing 'title' in prediction"
        assert 'confidence' in predictions[0], "Missing 'confidence' in prediction"
        
        print(f"✓ Generated {len(predictions)} predictions")
        print(f"  Sample: {predictions[0]['title']}")
        return True
    except Exception as e:
        print(f"✗ Article predictor error: {e}")
        return False

def test_book_recommender():
    """Test book recommendations"""
    print("\nTesting book recommender...")
    try:
        from data_collector import SubstackCollector
        from article_predictor import ArticlePredictor
        from book_recommender import BookRecommender
        
        # Get sample data
        collector = SubstackCollector("https://example.substack.com")
        articles_df = collector.collect_articles(max_articles=5)
        
        # Get predictions
        predictor = ArticlePredictor()
        predictor.fit(articles_df)
        predictions = predictor.predict_next_articles(num_predictions=3)
        
        # Test recommendations
        recommender = BookRecommender()
        books = recommender.recommend_books(articles_df, predictions, num_books=5)
        
        assert len(books) > 0, "No books recommended"
        assert 'title' in books[0], "Missing 'title' in recommendation"
        assert 'author' in books[0], "Missing 'author' in recommendation"
        
        print(f"✓ Generated {len(books)} book recommendations")
        print(f"  Sample: {books[0]['title']} by {books[0]['author']}")
        return True
    except Exception as e:
        print(f"✗ Book recommender error: {e}")
        return False

def test_visualizer():
    """Test visualization generation"""
    print("\nTesting visualizer...")
    try:
        from data_collector import SubstackCollector
        from article_predictor import ArticlePredictor
        from growth_visualizer import GrowthVisualizer
        
        # Get sample data
        collector = SubstackCollector("https://example.substack.com")
        articles_df = collector.collect_articles(max_articles=5)
        
        # Get predictions
        predictor = ArticlePredictor()
        predictor.fit(articles_df)
        predictions = predictor.predict_next_articles(num_predictions=3)
        
        # Create output directory
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Test visualization
        visualizer = GrowthVisualizer()
        visualizer.create_growth_timeline(
            articles_df,
            predictions,
            output_dir / "test_timeline.html"
        )
        
        assert (output_dir / "test_timeline.html").exists(), "Timeline file not created"
        
        print("✓ Visualization created successfully")
        
        # Cleanup
        import shutil
        shutil.rmtree(output_dir)
        
        return True
    except Exception as e:
        print(f"✗ Visualizer error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("SUBSTACK ANALYZER - TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_data_collector,
        test_article_predictor,
        test_book_recommender,
        test_visualizer
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! The analyzer is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python example_usage.py")
        print("  2. Or: python substack_analyzer.py --substack-url YOUR_URL")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nTip: Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
