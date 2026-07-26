"""
Test Dashboard Features - Demo Script
Demonstrates the core functionality without needing browser UI
"""

import sys
from pathlib import Path

print("="*70)
print("🎯 SUBSTACK ANALYZER - FEATURE DEMONSTRATION")
print("="*70)
print()

# Test 1: Data Collection
print("📥 TEST 1: Data Collection")
print("-" * 70)

from data_collector import SubstackCollector

substack_url = "https://substack.com/@ethannorton"
print(f"Collecting data from: {substack_url}")

collector = SubstackCollector(substack_url)
articles_df = collector.collect_articles(max_articles=10)

print(f"✅ Collected {len(articles_df)} articles")
print(f"\nSample Article:")
if len(articles_df) > 0:
    sample = articles_df.iloc[0]
    print(f"  Title: {sample['title']}")
    print(f"  Date: {sample['published_date']}")
    print(f"  Length: {len(sample['content'])} characters")
print()

# Test 2: Mastery Scoring
print("="*70)
print("🎯 TEST 2: Mastery Scoring")
print("-" * 70)

from mastery_scorer import MasteryScorer

scorer = MasteryScorer()
mastery_results = scorer.analyze_mastery(articles_df, articles_df)

print(f"\n📊 OVERALL MASTERY SCORE: {mastery_results['overall_score']:.1f}/100")
print(f"🏆 MASTERY LEVEL: {mastery_results['mastery_level']}")
print()

print("📈 Score Breakdown:")
for metric, score in mastery_results['metrics'].items():
    bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
    print(f"  {metric:20s}: {bar} {score:.1f}/100")

print(f"\n💪 Top Strengths:")
for i, strength in enumerate(mastery_results['strengths'][:3], 1):
    print(f"  {i}. {strength}")

print(f"\n🎯 Growth Areas:")
for i, area in enumerate(mastery_results['growth_areas'][:3], 1):
    print(f"  {i}. {area}")

print(f"\n💡 Recommendations:")
for i, rec in enumerate(mastery_results['recommendations'][:3], 1):
    print(f"  {i}. {rec}")

# Test 3: Topic Predictions
print()
print("="*70)
print("🔮 TEST 3: Article Predictions")
print("-" * 70)

from article_predictor import ArticlePredictor

predictor = ArticlePredictor()
print("Training prediction model...")
predictor.fit(articles_df)

predictions = predictor.predict_next_articles(num_predictions=5)

print(f"\n✨ Generated {len(predictions)} article suggestions:\n")

for i, pred in enumerate(predictions, 1):
    confidence_emoji = "🟢" if pred['confidence'] > 0.75 else "🟡" if pred['confidence'] > 0.5 else "🟠"
    print(f"{confidence_emoji} {i}. {pred['title']}")
    print(f"   Confidence: {pred['confidence']:.0%} | Topics: {', '.join(pred['topics'][:5])}")
    print()

# Test 4: Book Recommendations
print("="*70)
print("📚 TEST 4: Book Recommendations")
print("-" * 70)

from book_recommender import BookRecommender

recommender = BookRecommender()
books = recommender.recommend_books(articles_df, predictions, num_books=8)

print(f"\n📖 Top {len(books)} Recommended Books:\n")

for i, book in enumerate(books, 1):
    relevance_bar = "⭐" * int(book['relevance_score'] * 5)
    print(f"{i}. {book['title']}")
    print(f"   by {book['author']}")
    print(f"   Relevance: {relevance_bar} ({book['relevance_score']:.0%})")
    print(f"   Topics: {', '.join(book['topics'][:3])}")
    print(f"   Why: {book['reason']}")
    print()

# Test 5: Topic Expertise
print("="*70)
print("🎓 TEST 5: Topic Expertise Analysis")
print("-" * 70)

topic_expertise = mastery_results.get('topic_expertise', {})
if topic_expertise:
    print("\n📊 Your Expertise Levels:\n")
    
    sorted_topics = sorted(topic_expertise.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for topic, score in sorted_topics:
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        level = "Master" if score > 80 else "Expert" if score > 60 else "Advanced" if score > 40 else "Developing"
        print(f"  {topic:20s}: {bar} {score:.1f} ({level})")

# Summary
print()
print("="*70)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*70)
print()
print("🌟 Key Features Demonstrated:")
print("  ✓ Data collection from Substack")
print("  ✓ Mastery scoring across 5 dimensions")
print("  ✓ AI-powered topic predictions")
print("  ✓ Personalized book recommendations")
print("  ✓ Topic expertise analysis")
print()
print("💻 To see the full interactive dashboard, run:")
print("   streamlit run dashboard.py")
print()
print("📊 Or view the command-line version:")
print("   python3 substack_analyzer.py --substack-url YOUR_URL")
print()
print("="*70)
