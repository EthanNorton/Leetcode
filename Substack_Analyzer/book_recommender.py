"""
Book Recommendation Module
Recommends books based on article topics and predicted future content
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import json
from collections import Counter
import re


class BookRecommender:
    """Recommends books based on article content and topics"""
    
    def __init__(self):
        # Curated book database organized by topics
        self.book_database = self._load_book_database()
    
    def recommend_books(
        self,
        articles_df: pd.DataFrame,
        predictions: List[Dict],
        num_books: int = 10
    ) -> List[Dict]:
        """
        Generate book recommendations
        
        Args:
            articles_df: Historical articles
            predictions: Predicted future articles
            num_books: Number of books to recommend
            
        Returns:
            List of book recommendations
        """
        # Extract topics from articles and predictions
        article_topics = self._extract_topics_from_articles(articles_df)
        prediction_topics = self._extract_topics_from_predictions(predictions)
        
        # Combine and weight topics (predictions get higher weight)
        all_topics = article_topics + prediction_topics * 2
        topic_freq = Counter(all_topics)
        
        # Score books based on topic relevance
        book_scores = []
        
        for book in self.book_database:
            score = self._calculate_book_relevance(book, topic_freq)
            if score > 0:
                book_scores.append({
                    **book,
                    'relevance_score': score
                })
        
        # Sort by relevance and return top N
        book_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Add personalized reasons
        recommendations = []
        for book in book_scores[:num_books]:
            book_with_reason = book.copy()
            book_with_reason['reason'] = self._generate_recommendation_reason(
                book,
                topic_freq
            )
            recommendations.append(book_with_reason)
        
        return recommendations
    
    def _extract_topics_from_articles(self, articles_df: pd.DataFrame) -> List[str]:
        """Extract topic keywords from articles"""
        topics = []
        
        for _, article in articles_df.iterrows():
            text = f"{article.get('title', '')} {article.get('content', '')} {article.get('summary', '')}"
            tags = article.get('tags', [])
            
            if isinstance(tags, list):
                topics.extend([tag.lower() for tag in tags])
            
            # Extract key terms
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            topics.extend(words)
        
        return topics
    
    def _extract_topics_from_predictions(self, predictions: List[Dict]) -> List[str]:
        """Extract topics from predictions"""
        topics = []
        
        for pred in predictions:
            pred_topics = pred.get('topics', [])
            if isinstance(pred_topics, list):
                topics.extend([t.lower() for t in pred_topics])
            
            # Also extract from title and description
            text = f"{pred.get('title', '')} {pred.get('description', '')}"
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            topics.extend(words)
        
        return topics
    
    def _calculate_book_relevance(self, book: Dict, topic_freq: Counter) -> float:
        """Calculate how relevant a book is to the topics"""
        score = 0.0
        
        book_topics = book['topics']
        book_keywords = book['keywords']
        
        # Score based on topic matches
        for topic in book_topics:
            topic_lower = topic.lower()
            if topic_lower in topic_freq:
                score += topic_freq[topic_lower] * 2.0
        
        # Score based on keyword matches
        for keyword in book_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in topic_freq:
                score += topic_freq[keyword_lower] * 1.0
        
        # Normalize
        max_possible = sum(topic_freq.values())
        if max_possible > 0:
            score = score / max_possible
        
        return min(score, 1.0)
    
    def _generate_recommendation_reason(
        self,
        book: Dict,
        topic_freq: Counter
    ) -> str:
        """Generate explanation for why this book is recommended"""
        matching_topics = [
            topic for topic in book['topics']
            if topic.lower() in topic_freq
        ]
        
        if matching_topics:
            return f"Recommended based on your interest in {', '.join(matching_topics[:3])}"
        else:
            return f"Recommended to expand knowledge in {book['topics'][0]}"
    
    def save_recommendations(self, recommendations: List[Dict], output_path: str):
        """Save recommendations to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
    
    def _load_book_database(self) -> List[Dict]:
        """Load book database with topics"""
        # Comprehensive book database organized by topics
        return [
            # AI & Machine Learning
            {
                'title': 'Deep Learning',
                'author': 'Ian Goodfellow, Yoshua Bengio, Aaron Courville',
                'topics': ['AI', 'Machine Learning', 'Deep Learning', 'Neural Networks'],
                'keywords': ['artificial intelligence', 'neural', 'learning', 'algorithms', 'optimization'],
                'description': 'Comprehensive textbook on deep learning theory and practice'
            },
            {
                'title': 'Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow',
                'author': 'Aurélien Géron',
                'topics': ['Machine Learning', 'Python', 'Practical AI'],
                'keywords': ['machine learning', 'python', 'scikit-learn', 'tensorflow', 'practical'],
                'description': 'Practical guide to implementing ML systems'
            },
            {
                'title': 'Pattern Recognition and Machine Learning',
                'author': 'Christopher Bishop',
                'topics': ['Machine Learning', 'Statistics', 'Pattern Recognition'],
                'keywords': ['pattern recognition', 'bayesian', 'statistical', 'learning', 'models'],
                'description': 'Mathematical foundations of machine learning'
            },
            {
                'title': 'Artificial Intelligence: A Modern Approach',
                'author': 'Stuart Russell, Peter Norvig',
                'topics': ['AI', 'Computer Science', 'Algorithms'],
                'keywords': ['artificial intelligence', 'algorithms', 'search', 'logic', 'reasoning'],
                'description': 'Comprehensive introduction to AI concepts and techniques'
            },
            
            # NLP & Language
            {
                'title': 'Speech and Language Processing',
                'author': 'Dan Jurafsky, James H. Martin',
                'topics': ['NLP', 'Language', 'Computational Linguistics'],
                'keywords': ['natural language', 'processing', 'text', 'language models', 'linguistics'],
                'description': 'Comprehensive guide to NLP and computational linguistics'
            },
            {
                'title': 'Natural Language Processing with Python',
                'author': "Steven Bird, Ewan Klein, Edward Loper",
                'topics': ['NLP', 'Python', 'Text Processing'],
                'keywords': ['natural language', 'python', 'nltk', 'text', 'processing'],
                'description': 'Practical introduction to NLP with Python'
            },
            {
                'title': 'Neural Network Methods for Natural Language Processing',
                'author': 'Yoav Goldberg',
                'topics': ['NLP', 'Neural Networks', 'Deep Learning'],
                'keywords': ['neural', 'language', 'embeddings', 'rnn', 'attention'],
                'description': 'Deep learning approaches to NLP'
            },
            
            # Computer Vision
            {
                'title': 'Computer Vision: Algorithms and Applications',
                'author': 'Richard Szeliski',
                'topics': ['Computer Vision', 'Image Processing', 'Visual Recognition'],
                'keywords': ['computer vision', 'image', 'recognition', 'visual', 'processing'],
                'description': 'Comprehensive guide to computer vision algorithms'
            },
            {
                'title': 'Deep Learning for Vision Systems',
                'author': 'Mohamed Elgendy',
                'topics': ['Computer Vision', 'Deep Learning', 'CNN'],
                'keywords': ['vision', 'convolutional', 'image recognition', 'cnn', 'visual'],
                'description': 'Modern deep learning approaches to computer vision'
            },
            
            # Data Science
            {
                'title': 'Python for Data Analysis',
                'author': 'Wes McKinney',
                'topics': ['Data Science', 'Python', 'Pandas'],
                'keywords': ['data analysis', 'pandas', 'python', 'numpy', 'data manipulation'],
                'description': 'Essential guide to data analysis with Python'
            },
            {
                'title': 'The Data Science Handbook',
                'author': 'Field Cady',
                'topics': ['Data Science', 'Statistics', 'Analytics'],
                'keywords': ['data science', 'statistics', 'analytics', 'modeling', 'prediction'],
                'description': 'Comprehensive overview of data science practices'
            },
            {
                'title': 'Storytelling with Data',
                'author': 'Cole Nussbaumer Knaflic',
                'topics': ['Data Visualization', 'Communication', 'Analytics'],
                'keywords': ['visualization', 'communication', 'data', 'storytelling', 'presentation'],
                'description': 'Effective data visualization and communication'
            },
            
            # Ethics & Society
            {
                'title': 'Weapons of Math Destruction',
                'author': "Cathy O'Neil",
                'topics': ['Ethics', 'AI', 'Society', 'Algorithms'],
                'keywords': ['ethics', 'bias', 'fairness', 'algorithms', 'social impact'],
                'description': 'How algorithms can increase inequality'
            },
            {
                'title': 'The Alignment Problem',
                'author': 'Brian Christian',
                'topics': ['AI Ethics', 'AI Safety', 'Machine Learning'],
                'keywords': ['alignment', 'ethics', 'safety', 'values', 'control'],
                'description': 'AI alignment and safety challenges'
            },
            {
                'title': 'Atlas of AI',
                'author': 'Kate Crawford',
                'topics': ['AI', 'Ethics', 'Society', 'Politics'],
                'keywords': ['artificial intelligence', 'ethics', 'power', 'politics', 'social'],
                'description': 'Political and social dimensions of AI'
            },
            
            # Mathematics & Theory
            {
                'title': 'Mathematics for Machine Learning',
                'author': 'Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong',
                'topics': ['Mathematics', 'Machine Learning', 'Linear Algebra'],
                'keywords': ['mathematics', 'linear algebra', 'calculus', 'probability', 'optimization'],
                'description': 'Mathematical foundations for ML'
            },
            {
                'title': 'The Elements of Statistical Learning',
                'author': 'Trevor Hastie, Robert Tibshirani, Jerome Friedman',
                'topics': ['Statistics', 'Machine Learning', 'Data Mining'],
                'keywords': ['statistical learning', 'regression', 'classification', 'data mining', 'prediction'],
                'description': 'Statistical learning theory and methods'
            },
            {
                'title': 'Information Theory, Inference, and Learning Algorithms',
                'author': 'David MacKay',
                'topics': ['Information Theory', 'Bayesian', 'Machine Learning'],
                'keywords': ['information theory', 'bayesian', 'inference', 'coding', 'probability'],
                'description': 'Information theory and Bayesian inference'
            },
            
            # Programming & Software
            {
                'title': 'Designing Data-Intensive Applications',
                'author': 'Martin Kleppmann',
                'topics': ['Software Engineering', 'Distributed Systems', 'Databases'],
                'keywords': ['distributed', 'databases', 'scalability', 'systems', 'architecture'],
                'description': 'Building scalable data systems'
            },
            {
                'title': 'Clean Code',
                'author': 'Robert C. Martin',
                'topics': ['Software Engineering', 'Programming', 'Best Practices'],
                'keywords': ['clean code', 'programming', 'software', 'quality', 'practices'],
                'description': 'Writing maintainable and clean code'
            },
            
            # Reinforcement Learning
            {
                'title': 'Reinforcement Learning: An Introduction',
                'author': 'Richard S. Sutton, Andrew G. Barto',
                'topics': ['Reinforcement Learning', 'Machine Learning', 'AI'],
                'keywords': ['reinforcement', 'learning', 'agents', 'rewards', 'policy'],
                'description': 'Foundational text on reinforcement learning'
            },
            {
                'title': 'Deep Reinforcement Learning Hands-On',
                'author': 'Maxim Lapan',
                'topics': ['Reinforcement Learning', 'Deep Learning', 'Python'],
                'keywords': ['deep reinforcement', 'dqn', 'policy gradient', 'actor-critic', 'practical'],
                'description': 'Practical guide to deep RL'
            },
            
            # Productivity & Learning
            {
                'title': 'A Mind for Numbers',
                'author': 'Barbara Oakley',
                'topics': ['Learning', 'Education', 'Productivity'],
                'keywords': ['learning', 'mathematics', 'studying', 'memory', 'understanding'],
                'description': 'How to learn difficult technical subjects'
            },
            {
                'title': 'Deep Work',
                'author': 'Cal Newport',
                'topics': ['Productivity', 'Focus', 'Career'],
                'keywords': ['productivity', 'focus', 'concentration', 'work', 'success'],
                'description': 'Rules for focused success in a distracted world'
            },
            {
                'title': 'The Pragmatic Programmer',
                'author': 'David Thomas, Andrew Hunt',
                'topics': ['Programming', 'Software Engineering', 'Career'],
                'keywords': ['programming', 'software', 'career', 'practices', 'development'],
                'description': 'Your journey to mastery in programming'
            },
        ]
