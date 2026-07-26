"""
Mastery Scoring Module
Scores writing mastery based on consistency, depth, diversity, and growth
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from collections import Counter
from datetime import datetime, timedelta
import re


class MasteryScorer:
    """Calculates mastery scores for content creators"""
    
    def __init__(self):
        self.mastery_levels = {
            (0, 20): "Beginner",
            (20, 40): "Developing",
            (40, 60): "Intermediate",
            (60, 75): "Advanced",
            (75, 85): "Expert",
            (85, 100): "Master"
        }
    
    def analyze_mastery(self, current_df: pd.DataFrame, all_articles_df: pd.DataFrame) -> Dict:
        """
        Analyze writing mastery across multiple dimensions
        
        Args:
            current_df: Current period articles (filtered by date)
            all_articles_df: All historical articles
            
        Returns:
            Dictionary with mastery analysis results
        """
        metrics = {
            'consistency_score': self._calculate_consistency(all_articles_df),
            'depth_score': self._calculate_depth(current_df),
            'diversity_score': self._calculate_diversity(current_df),
            'growth_score': self._calculate_growth(all_articles_df),
            'engagement_score': self._calculate_engagement(current_df)
        }
        
        # Calculate overall score (weighted average)
        overall_score = (
            metrics['consistency_score'] * 0.25 +
            metrics['depth_score'] * 0.20 +
            metrics['diversity_score'] * 0.20 +
            metrics['growth_score'] * 0.20 +
            metrics['engagement_score'] * 0.15
        )
        
        # Determine mastery level
        mastery_level = self._get_mastery_level(overall_score)
        
        # Generate insights
        strengths = self._identify_strengths(metrics)
        growth_areas = self._identify_growth_areas(metrics)
        recommendations = self._generate_recommendations(metrics, mastery_level)
        
        # Topic expertise
        topic_expertise = self._calculate_topic_expertise(all_articles_df)
        
        return {
            'overall_score': round(overall_score, 1),
            'mastery_level': mastery_level,
            'metrics': metrics,
            'strengths': strengths,
            'growth_areas': growth_areas,
            'recommendations': recommendations,
            'topic_expertise': topic_expertise
        }
    
    def _calculate_consistency(self, articles_df: pd.DataFrame) -> float:
        """Calculate publishing consistency score"""
        if len(articles_df) < 2:
            return 50.0
        
        articles_df = articles_df.sort_values('published_date')
        
        # Calculate gaps between articles
        dates = pd.to_datetime(articles_df['published_date']).dropna()
        gaps = dates.diff().dt.days.dropna()
        
        if len(gaps) == 0:
            return 50.0
        
        # Ideal is publishing regularly (e.g., weekly = 7 days)
        avg_gap = gaps.mean()
        std_gap = gaps.std()
        
        # Score based on regularity (lower std is better)
        consistency_score = 100 - min(std_gap / avg_gap * 50, 50) if avg_gap > 0 else 50
        
        # Bonus for recent activity
        days_since_last = (datetime.now() - dates.max()).days
        recency_bonus = max(0, 20 - days_since_last / 7) if days_since_last < 140 else 0
        
        return min(100, consistency_score + recency_bonus)
    
    def _calculate_depth(self, articles_df: pd.DataFrame) -> float:
        """Calculate content depth score"""
        if articles_df.empty:
            return 50.0
        
        # Calculate average content length
        articles_df['content_length'] = articles_df['content'].fillna('').str.len()
        articles_df['word_count'] = articles_df['content'].fillna('').str.split().str.len()
        
        avg_length = articles_df['content_length'].mean()
        avg_words = articles_df['word_count'].mean()
        
        # Score based on article length (assuming longer = more depth)
        # 1000 words = 70 points, scales up
        depth_score = min(100, 30 + (avg_words / 1000) * 40)
        
        # Bonus for consistent depth
        length_std = articles_df['content_length'].std()
        length_mean = articles_df['content_length'].mean()
        if length_mean > 0:
            consistency_bonus = max(0, 20 - (length_std / length_mean) * 10)
            depth_score += consistency_bonus
        
        return min(100, depth_score)
    
    def _calculate_diversity(self, articles_df: pd.DataFrame) -> float:
        """Calculate topic diversity score"""
        if articles_df.empty:
            return 50.0
        
        # Extract keywords from titles and content
        all_words = []
        for _, article in articles_df.iterrows():
            text = f"{article.get('title', '')} {article.get('content', '')} {article.get('summary', '')}"
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            all_words.extend(words)
        
        # Calculate diversity metrics
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        if total_words == 0:
            return 50.0
        
        # Diversity ratio
        diversity_ratio = unique_words / total_words if total_words > 0 else 0
        
        # Count unique topics (simple keyword clustering)
        word_freq = Counter(all_words)
        top_words = [word for word, count in word_freq.most_common(50)]
        
        # Score based on vocabulary richness
        diversity_score = min(100, diversity_ratio * 200 + len(top_words))
        
        return diversity_score
    
    def _calculate_growth(self, articles_df: pd.DataFrame) -> float:
        """Calculate growth trajectory score"""
        if len(articles_df) < 3:
            return 50.0
        
        articles_df = articles_df.sort_values('published_date')
        
        # Calculate article length over time
        articles_df['word_count'] = articles_df['content'].fillna('').str.split().str.len()
        articles_df['index'] = range(len(articles_df))
        
        # Simple linear regression for growth trend
        if len(articles_df) >= 3:
            x = articles_df['index'].values
            y = articles_df['word_count'].values
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Positive slope = growth
            growth_score = 50 + min(50, slope / 10)
        else:
            growth_score = 50.0
        
        # Bonus for increasing frequency
        articles_df['year_month'] = pd.to_datetime(articles_df['published_date']).dt.to_period('M')
        monthly_counts = articles_df.groupby('year_month').size()
        
        if len(monthly_counts) >= 3:
            recent_avg = monthly_counts.tail(3).mean()
            early_avg = monthly_counts.head(3).mean()
            
            if early_avg > 0:
                frequency_growth = (recent_avg - early_avg) / early_avg * 20
                growth_score += max(-10, min(20, frequency_growth))
        
        return max(0, min(100, growth_score))
    
    def _calculate_engagement(self, articles_df: pd.DataFrame) -> float:
        """Calculate engagement proxy score"""
        if articles_df.empty:
            return 50.0
        
        # Proxy metrics for engagement
        # 1. Title effectiveness (question marks, numbers, power words)
        title_scores = []
        
        power_words = {
            'how', 'why', 'what', 'guide', 'ultimate', 'complete', 
            'essential', 'critical', 'important', 'secret', 'proven'
        }
        
        for title in articles_df['title'].fillna(''):
            score = 50
            title_lower = title.lower()
            
            # Questions tend to engage
            if '?' in title:
                score += 10
            
            # Numbers in titles
            if any(char.isdigit() for char in title):
                score += 10
            
            # Power words
            if any(word in title_lower for word in power_words):
                score += 15
            
            # Length sweet spot (6-12 words)
            word_count = len(title.split())
            if 6 <= word_count <= 12:
                score += 15
            
            title_scores.append(min(100, score))
        
        avg_title_score = np.mean(title_scores)
        
        # 2. Content structure (paragraphs, formatting)
        structure_scores = []
        
        for content in articles_df['content'].fillna(''):
            score = 50
            
            # Check for paragraph breaks
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 3:
                score += 20
            
            # Check for lists or bullet points
            if '-' in content or '•' in content or any(f'{i}.' in content for i in range(1, 10)):
                score += 15
            
            # Check for headers or sections
            if '#' in content or any(word.isupper() for word in content.split()[:10]):
                score += 15
            
            structure_scores.append(min(100, score))
        
        avg_structure_score = np.mean(structure_scores)
        
        # Combined engagement score
        engagement_score = (avg_title_score * 0.6 + avg_structure_score * 0.4)
        
        return min(100, engagement_score)
    
    def _get_mastery_level(self, score: float) -> str:
        """Determine mastery level from score"""
        for (min_score, max_score), level in self.mastery_levels.items():
            if min_score <= score < max_score:
                return level
        return "Master"
    
    def _identify_strengths(self, metrics: Dict) -> List[str]:
        """Identify top strengths based on metrics"""
        strengths = []
        
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        
        strength_messages = {
            'consistency_score': 'Strong publishing consistency and regular output',
            'depth_score': 'High-quality, in-depth content with substantial analysis',
            'diversity_score': 'Diverse topics and broad vocabulary range',
            'growth_score': 'Clear growth trajectory and improving content',
            'engagement_score': 'Engaging titles and well-structured content'
        }
        
        for metric, score in sorted_metrics[:3]:
            if score >= 60:
                strengths.append(strength_messages.get(metric, f'{metric}: {score:.0f}'))
        
        return strengths if strengths else ['Building foundation for future growth']
    
    def _identify_growth_areas(self, metrics: Dict) -> List[str]:
        """Identify areas for improvement"""
        growth_areas = []
        
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1])
        
        growth_messages = {
            'consistency_score': 'Improve publishing consistency and frequency',
            'depth_score': 'Increase content depth and article length',
            'diversity_score': 'Explore more diverse topics and expand vocabulary',
            'growth_score': 'Focus on progressive improvement over time',
            'engagement_score': 'Enhance title effectiveness and content structure'
        }
        
        for metric, score in sorted_metrics[:3]:
            if score < 70:
                growth_areas.append(growth_messages.get(metric, f'{metric}: {score:.0f}'))
        
        return growth_areas if growth_areas else ['Maintain current excellence']
    
    def _generate_recommendations(self, metrics: Dict, level: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if metrics['consistency_score'] < 60:
            recommendations.append('Set a regular publishing schedule (e.g., weekly or bi-weekly)')
        
        if metrics['depth_score'] < 60:
            recommendations.append('Aim for longer-form content (1500+ words) with deeper analysis')
        
        if metrics['diversity_score'] < 60:
            recommendations.append('Explore new topics adjacent to your core themes')
        
        if metrics['engagement_score'] < 60:
            recommendations.append('Use more engaging titles with numbers, questions, or power words')
        
        if level in ['Beginner', 'Developing']:
            recommendations.append('Focus on building consistent publishing habits first')
        elif level in ['Intermediate', 'Advanced']:
            recommendations.append('Develop your unique voice and deepen expertise in core topics')
        else:
            recommendations.append('Consider creating comprehensive guides or a content series')
        
        return recommendations[:5]
    
    def _calculate_topic_expertise(self, articles_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate expertise level for each topic"""
        topic_counts = Counter()
        topic_quality = {}
        
        for _, article in articles_df.iterrows():
            # Extract topics from content
            text = f"{article.get('title', '')} {article.get('content', '')} {article.get('summary', '')}"
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            
            # Get significant words
            word_freq = Counter(words)
            top_words = [word for word, count in word_freq.most_common(20)]
            
            for word in top_words:
                topic_counts[word] += 1
                
                # Quality based on content length
                quality = min(100, len(article.get('content', '')) / 50)
                if word not in topic_quality:
                    topic_quality[word] = []
                topic_quality[word].append(quality)
        
        # Calculate expertise scores
        expertise = {}
        for topic, count in topic_counts.most_common(15):
            # Combine frequency and average quality
            avg_quality = np.mean(topic_quality[topic])
            expertise_score = min(100, (count * 10) + (avg_quality * 0.3))
            expertise[topic.title()] = round(expertise_score, 1)
        
        return expertise
