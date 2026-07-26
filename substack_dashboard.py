"""
Substack Dashboard with 3-Month Period Selection
Interactive dashboard for viewing Substack posts by quarter
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from substack_fetcher import SubstackFetcher
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Substack Content Dashboard",
    page_icon="📰",
    layout="wide"
)


def main():
    st.title("📰 Substack Content Dashboard")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        username = st.text_input(
            "Substack Username",
            value="ethanmorton",
            help="Enter the Substack username (e.g., ethanmorton)"
        )
        
        st.markdown("---")
        st.subheader("📅 Time Period Selection")
        st.markdown("**Select 3-Month Periods**")
        
        start_year = st.number_input(
            "Start Year",
            min_value=2020,
            max_value=datetime.now().year,
            value=2024
        )
        
        end_year = st.number_input(
            "End Year",
            min_value=start_year,
            max_value=datetime.now().year + 1,
            value=datetime.now().year
        )
    
    # Initialize fetcher
    if username:
        fetcher = SubstackFetcher(username)
        
        # Generate 3-month periods
        periods = fetcher.get_three_month_periods(start_year, end_year)
        
        # Period selection
        period_labels = [p['label'] for p in periods]
        
        selected_periods = st.multiselect(
            "Select 3-Month Periods to View",
            options=period_labels,
            default=[period_labels[-1]] if period_labels else [],
            help="Select one or more 3-month periods"
        )
        
        if selected_periods:
            # Filter selected periods
            selected_period_objs = [p for p in periods if p['label'] in selected_periods]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Selected Periods", len(selected_periods))
            
            # Fetch posts button
            if st.button("🔄 Load Substack Content", type="primary"):
                with st.spinner(f"Fetching posts from {username}.substack.com..."):
                    all_posts = []
                    
                    for period in selected_period_objs:
                        posts = fetcher.fetch_posts_by_date_range(
                            period['start_date'],
                            period['end_date']
                        )
                        
                        for post in posts:
                            post['period'] = period['label']
                            post['quarter'] = period['quarter']
                            post['year'] = period['year']
                        
                        all_posts.extend(posts)
                    
                    # Store in session state
                    st.session_state['posts'] = all_posts
                    st.session_state['loaded'] = True
                    st.success(f"✅ Loaded {len(all_posts)} posts!")
        
        # Display results
        if st.session_state.get('loaded', False) and st.session_state.get('posts'):
            posts = st.session_state['posts']
            
            if posts:
                st.markdown("---")
                st.header(f"📊 Content Overview ({len(posts)} posts)")
                
                # Create DataFrame
                df = pd.DataFrame(posts)
                
                # Convert post_date to datetime
                if 'post_date' in df.columns:
                    df['post_date'] = pd.to_datetime(df['post_date'])
                    df['date'] = df['post_date'].dt.date
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Posts", len(posts))
                
                with col2:
                    total_words = df['word_count'].sum() if 'word_count' in df.columns else 0
                    st.metric("Total Words", f"{total_words:,}")
                
                with col3:
                    avg_words = df['word_count'].mean() if 'word_count' in df.columns else 0
                    st.metric("Avg Words/Post", f"{avg_words:.0f}")
                
                with col4:
                    unique_periods = df['period'].nunique() if 'period' in df.columns else 0
                    st.metric("Periods", unique_periods)
                
                # Visualizations
                st.markdown("---")
                
                tab1, tab2, tab3 = st.tabs(["📈 Timeline", "📋 Post List", "📊 Analytics"])
                
                with tab1:
                    st.subheader("Posts by 3-Month Period")
                    
                    if 'period' in df.columns:
                        period_counts = df['period'].value_counts().sort_index()
                        
                        fig = px.bar(
                            x=period_counts.index,
                            y=period_counts.values,
                            labels={'x': '3-Month Period', 'y': 'Number of Posts'},
                            title='Posts Distribution by Quarter'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Word count over time
                        if 'word_count' in df.columns:
                            st.subheader("Word Count Trend")
                            period_words = df.groupby('period')['word_count'].sum()
                            
                            fig2 = px.line(
                                x=period_words.index,
                                y=period_words.values,
                                labels={'x': '3-Month Period', 'y': 'Total Words'},
                                title='Total Words per Quarter'
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                
                with tab2:
                    st.subheader("Post Details")
                    
                    # Filter by period
                    period_filter = st.selectbox(
                        "Filter by Period",
                        options=['All'] + list(df['period'].unique())
                    )
                    
                    filtered_df = df if period_filter == 'All' else df[df['period'] == period_filter]
                    
                    # Sort by date
                    filtered_df = filtered_df.sort_values('post_date', ascending=False)
                    
                    # Display posts
                    for idx, post in filtered_df.iterrows():
                        with st.expander(f"📝 {post['title']} ({post['date']})"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Period:** {post['period']}")
                                st.markdown(f"**Date:** {post['date']}")
                                
                                if post.get('subtitle'):
                                    st.markdown(f"*{post['subtitle']}*")
                                
                                if post.get('description'):
                                    st.markdown(post['description'])
                            
                            with col2:
                                st.markdown(f"**Type:** {post.get('type', 'N/A')}")
                                st.markdown(f"**Words:** {post.get('word_count', 0):,}")
                                
                                if post.get('canonical_url'):
                                    st.markdown(f"[🔗 Read Post]({post['canonical_url']})")
                
                with tab3:
                    st.subheader("Content Analytics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Post types distribution
                        if 'type' in df.columns:
                            st.markdown("**Post Types**")
                            type_counts = df['type'].value_counts()
                            
                            fig3 = px.pie(
                                values=type_counts.values,
                                names=type_counts.index,
                                title='Post Type Distribution'
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                    
                    with col2:
                        # Audience distribution
                        if 'audience' in df.columns:
                            st.markdown("**Audience Distribution**")
                            audience_counts = df['audience'].value_counts()
                            
                            fig4 = px.pie(
                                values=audience_counts.values,
                                names=audience_counts.index,
                                title='Audience Type Distribution'
                            )
                            st.plotly_chart(fig4, use_container_width=True)
                    
                    # Word count statistics
                    if 'word_count' in df.columns:
                        st.markdown("---")
                        st.subheader("Word Count Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Min", f"{df['word_count'].min():,}")
                        with col2:
                            st.metric("Max", f"{df['word_count'].max():,}")
                        with col3:
                            st.metric("Mean", f"{df['word_count'].mean():.0f}")
                        with col4:
                            st.metric("Median", f"{df['word_count'].median():.0f}")
                        
                        # Word count histogram
                        fig5 = px.histogram(
                            df,
                            x='word_count',
                            nbins=20,
                            title='Word Count Distribution',
                            labels={'word_count': 'Words per Post', 'count': 'Frequency'}
                        )
                        st.plotly_chart(fig5, use_container_width=True)
            
            else:
                st.info("No posts found for the selected periods.")
        
        elif not st.session_state.get('loaded', False):
            st.info("👆 Select periods and click 'Load Substack Content' to view posts")


if __name__ == "__main__":
    main()
