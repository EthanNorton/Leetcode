"""
Growth Visualization Module
Creates interactive visualizations and timelines of content growth areas
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter
import re


class GrowthVisualizer:
    """Creates visualizations for content growth and evolution"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_growth_timeline(
        self,
        articles_df: pd.DataFrame,
        predictions: List[Dict],
        output_path: str
    ):
        """
        Create interactive timeline showing content growth areas
        
        Args:
            articles_df: Historical articles
            predictions: Predicted future articles
            output_path: Path to save HTML file
        """
        # Extract topics over time
        timeline_data = self._prepare_timeline_data(articles_df, predictions)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Content Evolution Timeline', 'Topic Distribution Over Time'),
            row_heights=[0.6, 0.4],
            vertical_spacing=0.15
        )
        
        # Add timeline visualization
        self._add_timeline_trace(fig, timeline_data, row=1)
        
        # Add topic distribution
        self._add_topic_distribution(fig, timeline_data, row=2)
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Content Growth Areas Timeline",
            title_font_size=24,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Save
        fig.write_html(output_path)
        print(f"Growth timeline saved to {output_path}")
    
    def create_topic_evolution(
        self,
        articles_df: pd.DataFrame,
        output_path: str
    ):
        """
        Create visualization showing how topics have evolved
        
        Args:
            articles_df: Historical articles
            output_path: Path to save HTML file
        """
        # Extract topics and their evolution
        topic_data = self._extract_topic_evolution(articles_df)
        
        # Create interactive line chart
        fig = go.Figure()
        
        for topic, data in topic_data.items():
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['frequency'],
                mode='lines+markers',
                name=topic,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        # Add prediction zone
        if len(articles_df) > 0 and 'published_date' in articles_df.columns:
            last_date = pd.to_datetime(articles_df['published_date']).max()
            future_date = last_date + timedelta(days=90)
            
            fig.add_vrect(
                x0=last_date, x1=future_date,
                fillcolor="lightgray", opacity=0.3,
                layer="below", line_width=0,
                annotation_text="Prediction Zone",
                annotation_position="top left"
            )
        
        fig.update_layout(
            title="Topic Evolution and Growth Areas",
            xaxis_title="Date",
            yaxis_title="Topic Frequency",
            height=600,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        fig.write_html(output_path)
        print(f"Topic evolution saved to {output_path}")
    
    def create_wordcloud(
        self,
        articles_df: pd.DataFrame,
        output_path: str
    ):
        """
        Create word cloud visualization
        
        Args:
            articles_df: Historical articles
            output_path: Path to save image file
        """
        # Combine all text
        all_text = ' '.join(
            articles_df['title'].fillna('') + ' ' +
            articles_df['content'].fillna('') + ' ' +
            articles_df['summary'].fillna('')
        )
        
        # Clean text
        all_text = re.sub(r'[^a-zA-Z\s]', '', all_text.lower())
        
        # Create word cloud
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_text)
        
        # Plot
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Key Themes and Topics', fontsize=30, pad=20)
        plt.tight_layout(pad=0)
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Word cloud saved to {output_path}")
    
    def _prepare_timeline_data(
        self,
        articles_df: pd.DataFrame,
        predictions: List[Dict]
    ) -> Dict:
        """Prepare data for timeline visualization"""
        timeline_data = {
            'historical': [],
            'predicted': []
        }
        
        # Process historical articles
        for _, article in articles_df.iterrows():
            timeline_data['historical'].append({
                'title': article.get('title', 'Untitled'),
                'date': pd.to_datetime(article.get('published_date', datetime.now())),
                'type': 'historical',
                'topics': self._extract_keywords(
                    f"{article.get('title', '')} {article.get('summary', '')}"
                )
            })
        
        # Add predictions
        last_date = pd.to_datetime(articles_df['published_date']).max() if len(articles_df) > 0 else datetime.now()
        
        for i, pred in enumerate(predictions):
            pred_date = last_date + timedelta(days=30 * (i + 1))
            timeline_data['predicted'].append({
                'title': pred['title'],
                'date': pred_date,
                'type': 'predicted',
                'topics': pred.get('topics', [])[:5],
                'confidence': pred.get('confidence', 0.5)
            })
        
        return timeline_data
    
    def _add_timeline_trace(self, fig, timeline_data: Dict, row: int):
        """Add timeline trace to figure"""
        # Historical articles
        hist_dates = [item['date'] for item in timeline_data['historical']]
        hist_titles = [item['title'] for item in timeline_data['historical']]
        hist_y = list(range(len(hist_dates)))
        
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=hist_y,
                mode='markers',
                name='Published Articles',
                marker=dict(
                    size=12,
                    color='#3498db',
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                text=hist_titles,
                hovertemplate='<b>%{text}</b><br>Date: %{x}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # Predicted articles
        pred_dates = [item['date'] for item in timeline_data['predicted']]
        pred_titles = [item['title'] for item in timeline_data['predicted']]
        pred_confidence = [item['confidence'] for item in timeline_data['predicted']]
        pred_y = list(range(len(hist_dates), len(hist_dates) + len(pred_dates)))
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_y,
                mode='markers',
                name='Predicted Articles',
                marker=dict(
                    size=12,
                    color='#e74c3c',
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                text=[f"{title}<br>Confidence: {conf:.0%}" 
                      for title, conf in zip(pred_titles, pred_confidence)],
                hovertemplate='<b>%{text}</b><br>Date: %{x}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=row, col=1)
        fig.update_yaxes(title_text="Articles", row=row, col=1, showticklabels=False)
    
    def _add_topic_distribution(self, fig, timeline_data: Dict, row: int):
        """Add topic distribution to figure"""
        # Count topics
        topic_counts = Counter()
        
        for item in timeline_data['historical']:
            topic_counts.update(item['topics'])
        
        for item in timeline_data['predicted']:
            topic_counts.update(item['topics'])
        
        # Get top topics
        top_topics = topic_counts.most_common(10)
        
        if top_topics:
            topics, counts = zip(*top_topics)
            
            fig.add_trace(
                go.Bar(
                    x=list(topics),
                    y=list(counts),
                    marker_color='#2ecc71',
                    name='Topic Frequency'
                ),
                row=row, col=1
            )
        
        fig.update_xaxes(title_text="Topics", row=row, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Frequency", row=row, col=1)
    
    def _extract_topic_evolution(self, articles_df: pd.DataFrame) -> Dict:
        """Extract how topics have evolved over time"""
        if len(articles_df) == 0 or 'published_date' not in articles_df.columns:
            return {}
        
        articles_df = articles_df.copy()
        articles_df['published_date'] = pd.to_datetime(articles_df['published_date'])
        articles_df = articles_df.sort_values('published_date')
        
        # Extract topics for each article
        topics_over_time = {}
        
        for _, article in articles_df.iterrows():
            date = article['published_date']
            keywords = self._extract_keywords(
                f"{article.get('title', '')} {article.get('summary', '')} {article.get('content', '')}"
            )
            
            for keyword in keywords[:5]:  # Top 5 keywords per article
                if keyword not in topics_over_time:
                    topics_over_time[keyword] = {'dates': [], 'frequency': []}
                
                topics_over_time[keyword]['dates'].append(date)
        
        # Calculate cumulative frequency
        for topic, data in topics_over_time.items():
            data['dates'] = sorted(data['dates'])
            data['frequency'] = list(range(1, len(data['dates']) + 1))
        
        # Return only topics that appear multiple times
        return {k: v for k, v in topics_over_time.items() if len(v['dates']) >= 2}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Remove common words
        stopwords = {
            'this', 'that', 'with', 'from', 'have', 'will', 'been',
            'were', 'their', 'there', 'about', 'would', 'could', 'should',
            'these', 'those', 'which', 'what', 'when', 'where', 'article'
        }
        
        words = [w for w in words if w not in stopwords]
        
        # Count and return most common
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(10)]
