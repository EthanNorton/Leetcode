"""
Data Collection Module
Collects and parses articles from Substack newsletters
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from typing import List, Dict
import time
from tqdm import tqdm


class SubstackCollector:
    """Collects articles from a Substack newsletter"""
    
    def __init__(self, substack_url: str):
        """
        Initialize the collector
        
        Args:
            substack_url: Base URL of the Substack (e.g., https://example.substack.com)
        """
        self.substack_url = substack_url.rstrip('/')
        self.rss_url = f"{self.substack_url}/feed"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def collect_articles(self, max_articles: int = 50) -> pd.DataFrame:
        """
        Collect articles from the Substack
        
        Args:
            max_articles: Maximum number of articles to collect
            
        Returns:
            DataFrame with article data
        """
        print(f"Fetching articles from {self.rss_url}...")
        
        try:
            # Try RSS feed first
            feed = feedparser.parse(self.rss_url)
            
            if feed.entries:
                articles = self._parse_rss_feed(feed, max_articles)
            else:
                # Fallback to web scraping
                articles = self._scrape_archive(max_articles)
            
            if not articles:
                print("Warning: Using sample data as no articles could be fetched")
                articles = self._generate_sample_data()
            
            df = pd.DataFrame(articles)
            
            # Process and clean data
            df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
            df = df.sort_values('published_date', ascending=False).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error collecting articles: {e}")
            print("Using sample data instead...")
            return pd.DataFrame(self._generate_sample_data())
    
    def _parse_rss_feed(self, feed, max_articles: int) -> List[Dict]:
        """Parse articles from RSS feed"""
        articles = []
        
        for entry in tqdm(feed.entries[:max_articles], desc="Processing articles"):
            article = {
                'title': entry.get('title', ''),
                'url': entry.get('link', ''),
                'published_date': entry.get('published', ''),
                'summary': entry.get('summary', ''),
                'content': '',
                'tags': [tag.term for tag in entry.get('tags', [])]
            }
            
            # Fetch full content
            try:
                article['content'] = self._fetch_article_content(article['url'])
                time.sleep(0.5)  # Be nice to the server
            except Exception as e:
                print(f"Could not fetch content for {article['url']}: {e}")
                article['content'] = article['summary']
            
            articles.append(article)
        
        return articles
    
    def _fetch_article_content(self, url: str) -> str:
        """Fetch full article content from URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article content (Substack uses these classes)
            content_div = soup.find('div', class_='available-content')
            if not content_div:
                content_div = soup.find('div', class_='body')
            if not content_div:
                content_div = soup.find('article')
            
            if content_div:
                # Extract text from paragraphs
                paragraphs = content_div.find_all('p')
                return ' '.join([p.get_text().strip() for p in paragraphs])
            
            return ""
            
        except Exception as e:
            raise Exception(f"Failed to fetch content: {e}")
    
    def _scrape_archive(self, max_articles: int) -> List[Dict]:
        """Scrape articles from archive page"""
        articles = []
        archive_url = f"{self.substack_url}/archive"
        
        try:
            response = self.session.get(archive_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article links (this is a simplified version)
            article_links = soup.find_all('a', href=True, limit=max_articles)
            
            for link in tqdm(article_links[:max_articles], desc="Scraping articles"):
                article_url = link['href']
                if not article_url.startswith('http'):
                    article_url = self.substack_url + article_url
                
                try:
                    article_data = self._scrape_article(article_url)
                    if article_data:
                        articles.append(article_data)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error scraping {article_url}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error scraping archive: {e}")
        
        return articles
    
    def _scrape_article(self, url: str) -> Dict:
        """Scrape individual article"""
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract article data
        title = soup.find('h1')
        title = title.get_text().strip() if title else ""
        
        content = self._fetch_article_content(url)
        
        return {
            'title': title,
            'url': url,
            'published_date': datetime.now().isoformat(),
            'summary': content[:500] if content else "",
            'content': content,
            'tags': []
        }
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample data for demonstration"""
        sample_articles = [
            {
                'title': 'The Future of AI and Machine Learning',
                'url': f'{self.substack_url}/p/future-of-ai',
                'published_date': '2026-07-15',
                'summary': 'Exploring the latest developments in artificial intelligence and their impact on society.',
                'content': 'Artificial intelligence has transformed rapidly over the past year. Large language models have become more sophisticated, and we are seeing practical applications across industries. This article explores the key trends and what they mean for the future of technology and human society.',
                'tags': ['AI', 'Technology', 'Machine Learning']
            },
            {
                'title': 'Deep Dive into Neural Networks',
                'url': f'{self.substack_url}/p/neural-networks',
                'published_date': '2026-07-01',
                'summary': 'Understanding the architecture and training of modern neural networks.',
                'content': 'Neural networks are the backbone of modern AI systems. From CNNs to Transformers, understanding their architecture is crucial for anyone working in AI. We explore backpropagation, optimization techniques, and the latest architectural innovations.',
                'tags': ['Neural Networks', 'Deep Learning', 'AI']
            },
            {
                'title': 'Natural Language Processing Breakthroughs',
                'url': f'{self.substack_url}/p/nlp-breakthroughs',
                'published_date': '2026-06-15',
                'summary': 'Recent advances in NLP and language understanding.',
                'content': 'Natural language processing has seen remarkable breakthroughs. From better language models to more efficient training methods, we are getting closer to truly understanding human language. This article covers the latest research and practical applications.',
                'tags': ['NLP', 'Language Models', 'AI']
            },
            {
                'title': 'Ethics in AI Development',
                'url': f'{self.substack_url}/p/ai-ethics',
                'published_date': '2026-06-01',
                'summary': 'Examining ethical considerations in AI development and deployment.',
                'content': 'As AI systems become more powerful, ethical considerations become increasingly important. We discuss bias in AI, privacy concerns, transparency, and the responsibility of AI developers to create systems that benefit humanity.',
                'tags': ['Ethics', 'AI', 'Society']
            },
            {
                'title': 'Computer Vision and Image Recognition',
                'url': f'{self.substack_url}/p/computer-vision',
                'published_date': '2026-05-15',
                'summary': 'Advances in computer vision and practical applications.',
                'content': 'Computer vision has revolutionized how machines perceive the world. From medical imaging to autonomous vehicles, vision systems are becoming more accurate and efficient. We explore the latest techniques and real-world applications.',
                'tags': ['Computer Vision', 'Image Recognition', 'Deep Learning']
            },
        ]
        
        return sample_articles
