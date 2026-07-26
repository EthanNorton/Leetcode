"""
Article Prediction Module
Predicts future article topics based on historical content using NLP and ML techniques
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import json
import re
from collections import Counter


class ArticlePredictor:
    """Predicts future article topics using topic modeling and trend analysis"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1
        )
        self.lda_model = None
        self.nmf_model = None
        self.topics = []
        self.topic_trends = {}
        
    def fit(self, articles_df: pd.DataFrame):
        """
        Fit the prediction models on historical articles
        
        Args:
            articles_df: DataFrame with article data
        """
        # Combine title and content for analysis
        articles_df['full_text'] = (
            articles_df['title'].fillna('') + ' ' + 
            articles_df['content'].fillna('') + ' ' + 
            articles_df['summary'].fillna('')
        )
        
        self.articles_df = articles_df
        
        # Create TF-IDF matrix
        print("Creating TF-IDF features...")
        self.tfidf_matrix = self.vectorizer.fit_transform(articles_df['full_text'])
        
        # Topic modeling with LDA
        print("Performing topic modeling...")
        n_topics = min(10, len(articles_df) // 2) if len(articles_df) > 5 else 3
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        self.lda_topics = self.lda_model.fit_transform(self.tfidf_matrix)
        
        # NMF for alternative topic extraction
        self.nmf_model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=200
        )
        self.nmf_topics = self.nmf_model.fit_transform(self.tfidf_matrix)
        
        # Extract topic keywords
        self.topics = self._extract_topics(n_topics)
        
        # Analyze topic trends over time
        self._analyze_topic_trends(articles_df)
        
    def predict_next_articles(self, num_predictions: int = 5) -> List[Dict]:
        """
        Predict future article topics
        
        Args:
            num_predictions: Number of predictions to generate
            
        Returns:
            List of predicted articles with metadata
        """
        predictions = []
        
        # Get trending topics
        trending_topics = self._get_trending_topics()
        
        # Generate predictions based on trends
        for i in range(num_predictions):
            prediction = self._generate_prediction(i, trending_topics)
            predictions.append(prediction)
        
        return predictions
    
    def _extract_topics(self, n_topics: int) -> List[Dict]:
        """Extract topic keywords from models"""
        topics = []
        feature_names = self.vectorizer.get_feature_names_out()
        
        for topic_idx in range(n_topics):
            # Get top words from LDA
            lda_top_indices = self.lda_model.components_[topic_idx].argsort()[-10:][::-1]
            lda_keywords = [feature_names[i] for i in lda_top_indices]
            
            # Get top words from NMF
            nmf_top_indices = self.nmf_model.components_[topic_idx].argsort()[-10:][::-1]
            nmf_keywords = [feature_names[i] for i in nmf_top_indices]
            
            # Combine and deduplicate
            combined_keywords = list(dict.fromkeys(lda_keywords + nmf_keywords))[:15]
            
            topics.append({
                'id': topic_idx,
                'keywords': combined_keywords,
                'label': self._generate_topic_label(combined_keywords)
            })
        
        return topics
    
    def _generate_topic_label(self, keywords: List[str]) -> str:
        """Generate a human-readable label for a topic"""
        # Use top 3 keywords to create label
        return ' & '.join([kw.title() for kw in keywords[:3]])
    
    def _analyze_topic_trends(self, articles_df: pd.DataFrame):
        """Analyze how topics have evolved over time"""
        if 'published_date' not in articles_df.columns:
            return
        
        articles_df = articles_df.copy()
        articles_df = articles_df.sort_values('published_date')
        
        # Assign dominant topic to each article
        dominant_topics = np.argmax(self.lda_topics, axis=1)
        articles_df['dominant_topic'] = dominant_topics
        
        # Calculate topic frequency over time
        for topic_id in range(len(self.topics)):
            topic_articles = articles_df[articles_df['dominant_topic'] == topic_id]
            
            if len(topic_articles) > 0:
                # Calculate trend (increasing/decreasing)
                recent_count = len(topic_articles.tail(len(topic_articles) // 3))
                early_count = len(topic_articles.head(len(topic_articles) // 3))
                
                trend_direction = "increasing" if recent_count > early_count else "decreasing"
                
                self.topic_trends[topic_id] = {
                    'frequency': len(topic_articles),
                    'trend': trend_direction,
                    'recent_score': recent_count / max(len(topic_articles) // 3, 1)
                }
    
    def _get_trending_topics(self) -> List[int]:
        """Get topics that are trending upward"""
        trending = []
        
        for topic_id, trend_data in self.topic_trends.items():
            if trend_data['trend'] == 'increasing':
                trending.append((topic_id, trend_data['recent_score']))
        
        # Sort by recent score
        trending.sort(key=lambda x: x[1], reverse=True)
        
        return [t[0] for t in trending]
    
    def _generate_prediction(self, prediction_idx: int, trending_topics: List[int]) -> Dict:
        """Generate a single article prediction"""
        # Select topic for this prediction
        if trending_topics and prediction_idx < len(trending_topics):
            topic_id = trending_topics[prediction_idx]
        else:
            # Fall back to random topic
            topic_id = prediction_idx % len(self.topics)
        
        topic = self.topics[topic_id]
        keywords = topic['keywords']
        
        # Generate title based on topic keywords
        title = self._generate_title(keywords, prediction_idx)
        
        # Calculate confidence based on trend strength
        if topic_id in self.topic_trends:
            confidence = min(0.95, 0.5 + self.topic_trends[topic_id]['recent_score'] / 2)
        else:
            confidence = 0.6
        
        # Generate description
        description = self._generate_description(keywords)
        
        return {
            'title': title,
            'confidence': confidence,
            'topics': keywords,
            'topic_label': topic['label'],
            'description': description,
            'estimated_interest': 'High' if confidence > 0.75 else 'Medium'
        }
    
    def _generate_title(self, keywords: List[str], idx: int) -> str:
        """Generate a plausible article title"""
        templates = [
            f"Exploring {keywords[0].title()} in {keywords[1].title()}",
            f"The Future of {keywords[0].title()}: {keywords[1].title()} Perspective",
            f"Understanding {keywords[0].title()} Through {keywords[1].title()}",
            f"Deep Dive: {keywords[0].title()} and {keywords[1].title()}",
            f"Why {keywords[0].title()} Matters for {keywords[1].title()}",
            f"A Guide to {keywords[0].title()} in Modern {keywords[1].title()}",
            f"Rethinking {keywords[0].title()}: New Approaches in {keywords[1].title()}",
            f"The Intersection of {keywords[0].title()} and {keywords[1].title()}",
        ]
        
        return templates[idx % len(templates)]
    
    def _generate_description(self, keywords: List[str]) -> str:
        """Generate a description for the predicted article"""
        return (
            f"This article will likely explore themes around {', '.join(keywords[:3])}, "
            f"building on previous discussions of {', '.join(keywords[3:5])}. "
            f"It may also touch on {keywords[5] if len(keywords) > 5 else 'related topics'}."
        )
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """Save predictions to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
