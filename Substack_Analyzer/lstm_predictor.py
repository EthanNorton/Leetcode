"""
LSTM-Based Article Predictor with Time-Decay Weighting
Uses recurrent neural networks to predict future article topics with emphasis on recent content
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
from datetime import datetime, timedelta
from collections import Counter
import re

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class ArticleDataset(Dataset):
    """PyTorch Dataset for article sequences"""
    
    def __init__(self, sequences, labels, weights):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.weights = torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.weights[idx]


class ArticleLSTM(nn.Module):
    """LSTM model for article topic prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=50):
        super(ArticleLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Attention mechanism for focusing on recent content
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x, return_attention=False):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention (focus on recent time steps)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction layers
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        if return_attention:
            return out, attention_weights.squeeze(-1)
        return out


class LSTMArticlePredictor:
    """
    Advanced article predictor using LSTM with time-decay weighting
    Emphasizes recent articles more heavily in predictions
    """
    
    def __init__(self, sequence_length=5, decay_rate=0.9):
        """
        Args:
            sequence_length: Number of past articles to consider
            decay_rate: Time decay factor (0-1), higher = more emphasis on recent
        """
        self.sequence_length = sequence_length
        self.decay_rate = decay_rate
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.topics = []
        
    def calculate_time_weights(self, dates: pd.Series) -> np.ndarray:
        """
        Calculate time-decay weights for articles
        More recent articles get higher weights
        """
        dates = pd.to_datetime(dates)
        
        # Calculate days since most recent article
        most_recent = dates.max()
        days_ago = (most_recent - dates).dt.total_seconds() / (24 * 3600)
        
        # Apply exponential decay: weight = decay_rate ^ days_ago
        weights = self.decay_rate ** (days_ago / 30)  # Decay per month
        
        # Normalize weights to sum to len(dates)
        weights = weights / weights.sum() * len(dates)
        
        return weights.values
    
    def fit(self, articles_df: pd.DataFrame, train_model=True):
        """
        Fit the LSTM predictor on historical articles
        
        Args:
            articles_df: DataFrame with article data
            train_model: Whether to train the LSTM (requires PyTorch)
        """
        print("Preparing data for LSTM predictor...")
        
        # Prepare text data
        articles_df['full_text'] = (
            articles_df['title'].fillna('') + ' ' + 
            articles_df['content'].fillna('') + ' ' + 
            articles_df['summary'].fillna('')
        )
        
        # Calculate time weights
        if 'published_date' in articles_df.columns:
            self.time_weights = self.calculate_time_weights(articles_df['published_date'])
        else:
            self.time_weights = np.ones(len(articles_df))
        
        # Create TF-IDF features with time weighting
        self.tfidf_matrix = self.vectorizer.fit_transform(articles_df['full_text'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Apply time weights to TF-IDF matrix
        weighted_tfidf = self.tfidf_matrix.multiply(self.time_weights.reshape(-1, 1))
        
        # Store for sequence creation
        self.articles_df = articles_df.sort_values('published_date') if 'published_date' in articles_df.columns else articles_df
        self.features = weighted_tfidf.toarray()
        
        # Extract topics from weighted features
        self._extract_topics()
        
        # Train LSTM if available and requested
        if TORCH_AVAILABLE and train_model and len(articles_df) >= self.sequence_length + 1:
            print("Training LSTM model...")
            self._train_lstm()
        else:
            if not TORCH_AVAILABLE:
                print("PyTorch not available. Using statistical predictions only.")
            elif len(articles_df) < self.sequence_length + 1:
                print(f"Not enough articles for LSTM training (need at least {self.sequence_length + 1})")
            self.model = None
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        sequences = []
        labels = []
        weights = []
        
        for i in range(len(self.features) - self.sequence_length):
            seq = self.features[i:i + self.sequence_length]
            label = self.features[i + self.sequence_length]
            weight = self.time_weights[i + self.sequence_length]
            
            sequences.append(seq)
            labels.append(label)
            weights.append(weight)
        
        return np.array(sequences), np.array(labels), np.array(weights)
    
    def _train_lstm(self):
        """Train the LSTM model"""
        # Create sequences
        X, y, weights = self._create_sequences()
        
        if len(X) == 0:
            print("No sequences to train on")
            return
        
        # Create dataset and dataloader
        dataset = ArticleDataset(X, y, weights)
        dataloader = DataLoader(dataset, batch_size=min(8, len(X)), shuffle=True)
        
        # Initialize model
        input_size = X.shape[2]
        output_size = y.shape[1]
        self.model = ArticleLSTM(input_size, hidden_size=64, num_layers=2, output_size=output_size)
        
        # Training setup
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 50
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_X, batch_y, batch_weights in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Weighted loss (emphasize recent articles)
                loss = criterion(outputs, batch_y)
                weighted_loss = (loss * batch_weights.unsqueeze(1)).mean()
                
                # Backward pass
                weighted_loss.backward()
                optimizer.step()
                
                total_loss += weighted_loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
        
        self.model.eval()
        print("LSTM training complete!")
    
    def _extract_topics(self):
        """Extract topics from weighted TF-IDF"""
        # Get top features by weighted importance
        feature_importance = np.asarray(self.tfidf_matrix.multiply(
            self.time_weights.reshape(-1, 1)
        ).sum(axis=0)).flatten()
        
        top_indices = feature_importance.argsort()[-50:][::-1]
        
        # Create topic clusters
        n_topics = min(10, len(self.articles_df) // 2) if len(self.articles_df) > 5 else 3
        topics_per_cluster = len(top_indices) // n_topics
        
        self.topics = []
        for i in range(n_topics):
            start_idx = i * topics_per_cluster
            end_idx = start_idx + topics_per_cluster if i < n_topics - 1 else len(top_indices)
            
            topic_indices = top_indices[start_idx:end_idx]
            keywords = [self.feature_names[idx] for idx in topic_indices]
            
            self.topics.append({
                'id': i,
                'keywords': keywords[:15],
                'label': ' & '.join([kw.title() for kw in keywords[:3]]),
                'importance': float(feature_importance[topic_indices].mean())
            })
        
        # Sort by importance (recent emphasis)
        self.topics.sort(key=lambda x: x['importance'], reverse=True)
    
    def predict_next_articles(self, num_predictions: int = 5) -> List[Dict]:
        """
        Predict future article topics using LSTM or statistical methods
        
        Args:
            num_predictions: Number of predictions to generate
            
        Returns:
            List of predicted articles with metadata
        """
        predictions = []
        
        if self.model is not None and TORCH_AVAILABLE:
            predictions = self._predict_with_lstm(num_predictions)
        else:
            predictions = self._predict_statistical(num_predictions)
        
        return predictions
    
    def _predict_with_lstm(self, num_predictions: int) -> List[Dict]:
        """Generate predictions using trained LSTM"""
        predictions = []
        
        # Get recent sequence
        recent_sequence = self.features[-self.sequence_length:]
        recent_sequence = torch.FloatTensor(recent_sequence).unsqueeze(0)
        
        with torch.no_grad():
            # Get prediction with attention
            output, attention = self.model(recent_sequence, return_attention=True)
            predicted_features = output[0].numpy()
            attention_weights = attention[0].numpy()
        
        # Get top features from prediction
        top_feature_indices = predicted_features.argsort()[-30:][::-1]
        predicted_keywords = [self.feature_names[idx] for idx in top_feature_indices]
        
        # Generate predictions based on LSTM output and recent topics
        for i in range(num_predictions):
            # Mix LSTM prediction with trending topics
            if i < len(self.topics):
                base_topic = self.topics[i]
                # Blend LSTM keywords with topic keywords
                combined_keywords = list(dict.fromkeys(
                    predicted_keywords[:10] + base_topic['keywords'][:10]
                ))[:15]
            else:
                combined_keywords = predicted_keywords[:15]
            
            # Calculate confidence based on attention and topic importance
            base_confidence = 0.7 + (attention_weights[-3:].mean() * 0.25)  # Emphasize recent
            confidence = min(0.95, base_confidence - (i * 0.05))
            
            title = self._generate_title(combined_keywords, i)
            description = self._generate_description(combined_keywords)
            
            predictions.append({
                'title': title,
                'confidence': float(confidence),
                'topics': combined_keywords,
                'topic_label': ' & '.join([kw.title() for kw in combined_keywords[:3]]),
                'description': description,
                'estimated_interest': 'High' if confidence > 0.75 else 'Medium',
                'method': 'LSTM',
                'attention_score': float(attention_weights[-1])  # Most recent attention
            })
        
        return predictions
    
    def _predict_statistical(self, num_predictions: int) -> List[Dict]:
        """Fallback to statistical predictions"""
        predictions = []
        
        # Use weighted topic importance
        for i in range(min(num_predictions, len(self.topics))):
            topic = self.topics[i]
            keywords = topic['keywords']
            
            # Confidence based on importance and recency
            confidence = min(0.95, 0.5 + (topic['importance'] * 0.4))
            
            title = self._generate_title(keywords, i)
            description = self._generate_description(keywords)
            
            predictions.append({
                'title': title,
                'confidence': float(confidence),
                'topics': keywords,
                'topic_label': topic['label'],
                'description': description,
                'estimated_interest': 'High' if confidence > 0.75 else 'Medium',
                'method': 'Statistical (Time-Weighted)',
                'importance_score': float(topic['importance'])
            })
        
        return predictions
    
    def _generate_title(self, keywords: List[str], idx: int) -> str:
        """Generate article title from keywords"""
        templates = [
            f"The Future of {keywords[0].title()}: {keywords[1].title()} Insights",
            f"Understanding {keywords[0].title()} in Modern {keywords[1].title()}",
            f"Deep Dive: {keywords[0].title()} and {keywords[1].title()}",
            f"Why {keywords[0].title()} Matters for {keywords[1].title()}",
            f"Exploring {keywords[0].title()} Through {keywords[1].title()}",
            f"{keywords[0].title()} Revolution: The {keywords[1].title()} Perspective",
            f"Mastering {keywords[0].title()}: A {keywords[1].title()} Guide",
            f"From {keywords[0].title()} to {keywords[1].title()}: A Journey",
        ]
        
        return templates[idx % len(templates)]
    
    def _generate_description(self, keywords: List[str]) -> str:
        """Generate description for predicted article"""
        return (
            f"This article will explore {', '.join(keywords[:3])}, "
            f"with insights into {', '.join(keywords[3:5])}. "
            f"Drawing on recent developments in {keywords[0]}, "
            f"we'll examine how these concepts shape the future."
        )
    
    def get_topic_trends(self) -> pd.DataFrame:
        """Get topic trends with time weighting"""
        trend_data = []
        
        for topic in self.topics:
            trend_data.append({
                'Topic': topic['label'],
                'Importance': topic['importance'],
                'Keywords': ', '.join(topic['keywords'][:5]),
                'Trend': 'Rising' if topic['importance'] > 0.5 else 'Stable'
            })
        
        return pd.DataFrame(trend_data)
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """Save predictions to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
