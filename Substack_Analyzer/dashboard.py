"""
Interactive Substack Analyzer Dashboard
Provides monthly filtering, topic suggestions, mastery scoring, and book recommendations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import Counter
import numpy as np

from data_collector import SubstackCollector
from article_predictor import ArticlePredictor
from book_recommender import BookRecommender
from mastery_scorer import MasteryScorer
from growth_visualizer import GrowthVisualizer

# Page config
st.set_page_config(
    page_title="Substack Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .suggestion-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .book-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .mastery-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    .beginner { background: #e3f2fd; color: #1976d2; }
    .intermediate { background: #fff3e0; color: #f57c00; }
    .advanced { background: #e8f5e9; color: #388e3c; }
    .expert { background: #f3e5f5; color: #7b1fa2; }
    .master { background: #fce4ec; color: #c2185b; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_substack_data(substack_url, max_articles=100):
    """Load and cache Substack data"""
    collector = SubstackCollector(substack_url)
    articles_df = collector.collect_articles(max_articles=max_articles)
    return articles_df


@st.cache_data(ttl=3600)
def generate_predictions(_predictor, articles_df, num_predictions=10):
    """Generate and cache predictions"""
    _predictor.fit(articles_df)
    predictions = _predictor.predict_next_articles(num_predictions=num_predictions)
    return predictions


def get_mastery_badge_html(level):
    """Generate HTML for mastery badge"""
    badge_class = level.lower().replace(' ', '-')
    return f'<span class="mastery-badge {badge_class}">{level}</span>'


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Substack Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Substack URL input
    default_url = "https://substack.com/@ethannorton"
    substack_url = st.sidebar.text_input(
        "Substack URL",
        value=default_url,
        help="Enter your Substack profile or newsletter URL"
    )
    
    # Analysis parameters
    max_articles = st.sidebar.slider(
        "Maximum Articles to Analyze",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )
    
    num_predictions = st.sidebar.slider(
        "Number of Topic Suggestions",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    num_books = st.sidebar.slider(
        "Number of Book Recommendations",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )
    
    # Load data button
    if st.sidebar.button("🔄 Analyze Substack", type="primary"):
        st.session_state.data_loaded = False
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load and analyze data
    if not st.session_state.data_loaded:
        with st.spinner("📥 Loading your Substack data..."):
            articles_df = load_substack_data(substack_url, max_articles)
            
            if articles_df.empty:
                st.error("❌ No articles found. Please check your Substack URL.")
                return
            
            st.session_state.articles_df = articles_df
            st.session_state.data_loaded = True
            st.success(f"✅ Loaded {len(articles_df)} articles!")
    
    articles_df = st.session_state.articles_df
    
    # Date filtering
    st.sidebar.markdown("---")
    st.sidebar.header("📅 Date Filters")
    
    articles_df['published_date'] = pd.to_datetime(articles_df['published_date'])
    articles_df['year_month'] = articles_df['published_date'].dt.to_period('M')
    
    # Get unique months
    available_months = sorted(articles_df['year_month'].dropna().unique(), reverse=True)
    month_options = ['All Time'] + [str(m) for m in available_months]
    
    selected_month = st.sidebar.selectbox(
        "Filter by Month",
        options=month_options,
        index=0
    )
    
    # Filter data by month
    if selected_month != 'All Time':
        filtered_df = articles_df[articles_df['year_month'] == pd.Period(selected_month)]
        date_label = selected_month
    else:
        filtered_df = articles_df
        date_label = "All Time"
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎯 Mastery Score",
        "💡 Topic Suggestions",
        "📚 Book Recommendations",
        "📈 Growth Analytics"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header(f"📊 Overview - {date_label}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Articles",
                value=len(filtered_df),
                delta=f"{len(filtered_df) - len(articles_df[articles_df['published_date'] < filtered_df['published_date'].min()])} new" if selected_month != 'All Time' else None
            )
        
        with col2:
            avg_length = filtered_df['content'].str.len().mean()
            st.metric(
                label="Avg. Article Length",
                value=f"{int(avg_length):,} chars"
            )
        
        with col3:
            if selected_month != 'All Time':
                articles_this_month = len(filtered_df)
                st.metric(
                    label="Articles This Period",
                    value=articles_this_month
                )
            else:
                months_active = len(available_months)
                st.metric(
                    label="Months Active",
                    value=months_active
                )
        
        with col4:
            days_active = (articles_df['published_date'].max() - articles_df['published_date'].min()).days
            st.metric(
                label="Days Active",
                value=days_active
            )
        
        st.markdown("---")
        
        # Publishing frequency chart
        st.subheader("📅 Publishing Frequency")
        
        freq_df = articles_df.groupby('year_month').size().reset_index(name='count')
        freq_df['year_month'] = freq_df['year_month'].astype(str)
        
        fig = px.bar(
            freq_df,
            x='year_month',
            y='count',
            title='Articles Published per Month',
            labels={'year_month': 'Month', 'count': 'Number of Articles'},
            color='count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent articles
        st.subheader("📝 Recent Articles")
        recent_articles = filtered_df.head(10)
        
        for idx, row in recent_articles.iterrows():
            with st.expander(f"**{row['title']}** - {row['published_date'].strftime('%Y-%m-%d')}"):
                st.write(f"**Summary:** {row['summary'][:300]}...")
                if row.get('url'):
                    st.markdown(f"[Read Full Article]({row['url']})")
    
    # TAB 2: Mastery Score
    with tab2:
        st.header(f"🎯 Content Mastery Analysis - {date_label}")
        
        with st.spinner("Calculating mastery scores..."):
            scorer = MasteryScorer()
            mastery_results = scorer.analyze_mastery(filtered_df, articles_df)
        
        # Overall mastery score
        col1, col2 = st.columns([1, 2])
        
        with col1:
            overall_score = mastery_results['overall_score']
            level = mastery_results['mastery_level']
            
            # Circular progress
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Mastery Score", 'font': {'size': 24}},
                delta={'reference': 70},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 20], 'color': "#e3f2fd"},
                        {'range': [20, 40], 'color': "#bbdefb"},
                        {'range': [40, 60], 'color': "#90caf9"},
                        {'range': [60, 80], 'color': "#42a5f5"},
                        {'range': [80, 100], 'color': "#1976d2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"### Mastery Level")
            st.markdown(get_mastery_badge_html(level), unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Score Breakdown")
            
            metrics = mastery_results['metrics']
            
            breakdown_df = pd.DataFrame({
                'Metric': ['Consistency', 'Depth', 'Topic Diversity', 'Expertise Growth', 'Engagement'],
                'Score': [
                    metrics['consistency_score'],
                    metrics['depth_score'],
                    metrics['diversity_score'],
                    metrics['growth_score'],
                    metrics['engagement_score']
                ]
            })
            
            fig = px.bar(
                breakdown_df,
                x='Score',
                y='Metric',
                orientation='h',
                title='Mastery Components',
                color='Score',
                color_continuous_scale='Viridis',
                text='Score'
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(height=400, xaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("💪 Strengths")
            for strength in mastery_results['strengths'][:3]:
                st.success(f"✓ {strength}")
        
        with col2:
            st.subheader("🎯 Growth Areas")
            for area in mastery_results['growth_areas'][:3]:
                st.warning(f"→ {area}")
        
        with col3:
            st.subheader("📈 Recommendations")
            for rec in mastery_results['recommendations'][:3]:
                st.info(f"💡 {rec}")
        
        # Topic expertise breakdown
        st.markdown("---")
        st.subheader("🎓 Topic Expertise")
        
        topic_expertise = mastery_results.get('topic_expertise', {})
        if topic_expertise:
            expertise_df = pd.DataFrame([
                {'Topic': topic, 'Expertise': score}
                for topic, score in sorted(topic_expertise.items(), key=lambda x: x[1], reverse=True)[:10]
            ])
            
            fig = px.scatter(
                expertise_df,
                x='Topic',
                y='Expertise',
                size='Expertise',
                color='Expertise',
                title='Expertise by Topic',
                color_continuous_scale='Plasma',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Topic Suggestions
    with tab3:
        st.header(f"💡 Suggested Topics & Articles - {date_label}")
        
        st.markdown("""
        Based on your writing patterns and trending topics, here are personalized suggestions
        for your next articles. These predictions use machine learning to identify gaps and opportunities.
        """)
        
        with st.spinner("Generating topic suggestions..."):
            predictor = ArticlePredictor()
            predictions = generate_predictions(predictor, filtered_df, num_predictions)
        
        # Display as pacesetter timeline
        st.subheader("🎯 Your Content Roadmap")
        
        # Create timeline visualization
        last_date = filtered_df['published_date'].max()
        
        timeline_data = []
        for i, pred in enumerate(predictions):
            suggested_date = last_date + timedelta(days=7 * (i + 1))  # Weekly suggestions
            timeline_data.append({
                'Date': suggested_date,
                'Title': pred['title'],
                'Confidence': pred['confidence'],
                'Topics': ', '.join(pred['topics'][:5])
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Interactive timeline
        fig = go.Figure()
        
        # Historical articles
        fig.add_trace(go.Scatter(
            x=filtered_df['published_date'],
            y=[1] * len(filtered_df),
            mode='markers',
            name='Published',
            marker=dict(size=12, color='#1976d2', symbol='circle'),
            text=filtered_df['title'],
            hovertemplate='<b>%{text}</b><br>Date: %{x}<extra></extra>'
        ))
        
        # Suggested articles
        fig.add_trace(go.Scatter(
            x=timeline_df['Date'],
            y=[1] * len(timeline_df),
            mode='markers',
            name='Suggested',
            marker=dict(size=14, color='#e74c3c', symbol='diamond'),
            text=timeline_df['Title'],
            hovertemplate='<b>%{text}</b><br>Suggested: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Content Publishing Roadmap',
            xaxis_title='Date',
            yaxis=dict(showticklabels=False, showgrid=False),
            height=300,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed suggestions
        st.subheader("📝 Detailed Topic Suggestions")
        
        for i, pred in enumerate(predictions, 1):
            confidence_color = "🟢" if pred['confidence'] > 0.75 else "🟡" if pred['confidence'] > 0.5 else "🟠"
            
            with st.expander(f"{confidence_color} **{i}. {pred['title']}** (Confidence: {pred['confidence']:.0%})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {pred['description']}")
                    st.markdown(f"**Key Topics:** {', '.join(pred['topics'][:8])}")
                    st.markdown(f"**Interest Level:** {pred['estimated_interest']}")
                
                with col2:
                    st.metric("Confidence", f"{pred['confidence']:.0%}")
                    st.metric("Priority", f"#{i}")
                
                # Action buttons
                if st.button(f"📋 Copy Title", key=f"copy_{i}"):
                    st.code(pred['title'])
                    st.success("Title copied to display!")
    
    # TAB 4: Book Recommendations
    with tab4:
        st.header(f"📚 Personalized Book Recommendations - {date_label}")
        
        st.markdown("""
        Based on your content themes and predicted growth areas, here are books
        that will enhance your expertise and provide material for future articles.
        """)
        
        with st.spinner("Generating book recommendations..."):
            predictor = ArticlePredictor()
            predictions = generate_predictions(predictor, filtered_df, num_predictions)
            
            recommender = BookRecommender()
            books = recommender.recommend_books(filtered_df, predictions, num_books=num_books)
        
        # Filter options
        col1, col2 = st.columns([3, 1])
        with col1:
            topic_filter = st.multiselect(
                "Filter by Topics",
                options=list(set([topic for book in books for topic in book['topics']])),
                default=[]
            )
        
        with col2:
            min_relevance = st.slider(
                "Min. Relevance",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        # Filter books
        filtered_books = [
            book for book in books
            if book['relevance_score'] >= min_relevance and
            (not topic_filter or any(topic in book['topics'] for topic in topic_filter))
        ]
        
        st.markdown(f"### 📖 {len(filtered_books)} Recommended Books")
        
        # Display books in a grid
        cols = st.columns(2)
        
        for idx, book in enumerate(filtered_books):
            with cols[idx % 2]:
                relevance_bar = "🟩" * int(book['relevance_score'] * 5) + "⬜" * (5 - int(book['relevance_score'] * 5))
                
                st.markdown(f"""
                <div class="book-card">
                    <h4>{book['title']}</h4>
                    <p><strong>Author:</strong> {book['author']}</p>
                    <p><strong>Topics:</strong> {', '.join(book['topics'][:3])}</p>
                    <p><strong>Relevance:</strong> {relevance_bar} {book['relevance_score']:.0%}</p>
                    <p><em>{book['reason']}</em></p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🔖 Add to Reading List", key=f"book_{idx}"):
                    st.success(f"Added '{book['title']}' to your reading list!")
    
    # TAB 5: Growth Analytics
    with tab5:
        st.header(f"📈 Growth Analytics - {date_label}")
        
        # Topic evolution over time
        st.subheader("🔄 Topic Evolution")
        
        # Extract topics over time
        topics_over_time = {}
        for _, article in filtered_df.iterrows():
            date = article['published_date']
            text = f"{article['title']} {article['content']}"
            words = text.lower().split()
            
            # Count significant words
            for word in words:
                if len(word) > 4:
                    if word not in topics_over_time:
                        topics_over_time[word] = []
                    topics_over_time[word].append(date)
        
        # Get top topics
        top_topics = sorted(
            [(topic, len(dates)) for topic, dates in topics_over_time.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Create evolution chart
        fig = go.Figure()
        
        for topic, count in top_topics[:5]:
            dates = sorted(topics_over_time[topic])
            cumulative = list(range(1, len(dates) + 1))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative,
                mode='lines+markers',
                name=topic.title(),
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Topic Growth Over Time (Cumulative)',
            xaxis_title='Date',
            yaxis_title='Article Count',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Content depth analysis
        st.markdown("---")
        st.subheader("📏 Content Depth Analysis")
        
        filtered_df['word_count'] = filtered_df['content'].str.split().str.len()
        
        fig = px.histogram(
            filtered_df,
            x='word_count',
            nbins=20,
            title='Distribution of Article Lengths',
            labels={'word_count': 'Word Count', 'count': 'Number of Articles'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth trajectory
        st.markdown("---")
        st.subheader("📊 Writing Consistency")
        
        # Calculate rolling average
        freq_df = articles_df.groupby(articles_df['published_date'].dt.to_period('M')).size().reset_index(name='count')
        freq_df['published_date'] = freq_df['published_date'].astype(str)
        freq_df['rolling_avg'] = freq_df['count'].rolling(window=3, min_periods=1).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=freq_df['published_date'],
            y=freq_df['count'],
            name='Articles per Month',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            x=freq_df['published_date'],
            y=freq_df['rolling_avg'],
            name='3-Month Average',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title='Publishing Consistency Trend',
            xaxis_title='Month',
            yaxis_title='Number of Articles',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit | Substack Analyzer v1.0</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
