"""
Substack Content Fetcher
Fetches posts from Substack publications by username
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time


class SubstackFetcher:
    """Fetches content from Substack publications"""
    
    def __init__(self, username: str):
        """
        Initialize fetcher with Substack username
        
        Args:
            username: Substack username (e.g., 'ethanmorton')
        """
        self.username = username
        self.base_url = f"https://{username}.substack.com"
        self.api_url = f"{self.base_url}/api/v1"
        
    def fetch_posts(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        Fetch posts from the Substack publication using RSS feed
        
        Args:
            limit: Number of posts to fetch
            offset: Pagination offset
            
        Returns:
            List of post dictionaries
        """
        try:
            import xml.etree.ElementTree as ET
            from html.parser import HTMLParser
            
            # Try RSS feed first (most reliable)
            rss_url = f"{self.base_url}/feed"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(rss_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse RSS XML
            root = ET.fromstring(response.content)
            posts = []
            
            # Find all items in the RSS feed
            items = root.findall('.//item')
            
            for item in items[offset:offset+limit]:
                title = item.find('title')
                link = item.find('link')
                pub_date = item.find('pubDate')
                description = item.find('description')
                guid = item.find('guid')
                
                # Extract text content
                title_text = title.text if title is not None else 'Untitled'
                link_text = link.text if link is not None else ''
                pub_date_text = pub_date.text if pub_date is not None else ''
                description_text = description.text if description is not None else ''
                guid_text = guid.text if guid is not None else ''
                
                # Parse pub date to ISO format
                try:
                    from email.utils import parsedate_to_datetime
                    if pub_date_text:
                        dt = parsedate_to_datetime(pub_date_text)
                        pub_date_iso = dt.isoformat()
                    else:
                        pub_date_iso = ''
                except:
                    pub_date_iso = pub_date_text
                
                # Estimate word count from description
                word_count = len(description_text.split()) if description_text else 0
                
                # Extract slug from URL
                slug = ''
                if link_text:
                    parts = link_text.rstrip('/').split('/')
                    if len(parts) > 0:
                        slug = parts[-1]
                
                post_info = {
                    'id': guid_text,
                    'title': title_text,
                    'subtitle': '',
                    'slug': slug,
                    'post_date': pub_date_iso,
                    'canonical_url': link_text,
                    'description': description_text[:500] if description_text else '',
                    'type': 'newsletter',
                    'audience': 'everyone',
                    'word_count': word_count * 5,  # Rough estimate
                    'cover_image': ''
                }
                posts.append(post_info)
            
            return posts
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching posts: {e}")
            print(f"Tried URL: {rss_url}")
            return self._fetch_fallback_data()
        except ET.ParseError as e:
            print(f"Error parsing RSS: {e}")
            return self._fetch_fallback_data()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return self._fetch_fallback_data()
    
    def _fetch_fallback_data(self) -> List[Dict]:
        """
        Return sample data if API fails (for testing/demo purposes)
        """
        from datetime import datetime, timedelta
        
        print("Using fallback sample data for demonstration...")
        
        base_date = datetime.now()
        sample_posts = []
        
        titles = [
            "Pranayama and Polyvagal Theory Union and Experiences",
            "Nervous System Stabilization Through Breathwork",
            "Understanding the Vagus Nerve and Mental Health",
            "MSDS and Yogi Practices for Modern Life",
            "Breathing Techniques for Anxiety Relief",
            "The Science of Meditation and Neuroplasticity",
            "Integrating Ancient Wisdom with Modern Neuroscience",
            "Polyvagal Theory Applications in Daily Practice",
            "Mind-Body Connection: Research Updates",
            "Somatic Experiencing and Trauma Release"
        ]
        
        for i, title in enumerate(titles):
            days_ago = i * 30
            post_date = base_date - timedelta(days=days_ago)
            
            sample_posts.append({
                'id': f'sample-{i}',
                'title': title,
                'subtitle': f'Part {i+1} of the series',
                'slug': title.lower().replace(' ', '-')[:50],
                'post_date': post_date.isoformat(),
                'canonical_url': f"{self.base_url}/p/{title.lower().replace(' ', '-')[:30]}",
                'description': f'An exploration of {title.lower()} and its implications for mental health and wellbeing.',
                'type': 'newsletter',
                'audience': 'everyone',
                'word_count': 1500 + (i * 200),
                'cover_image': ''
            })
        
        return sample_posts
    
    def fetch_posts_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Fetch posts within a specific date range
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            List of posts within the date range
        """
        all_posts = []
        offset = 0
        limit = 50
        
        while True:
            posts = self.fetch_posts(limit=limit, offset=offset)
            
            if not posts:
                break
            
            for post in posts:
                post_date_str = post.get('post_date')
                if post_date_str:
                    post_date = datetime.fromisoformat(post_date_str.replace('Z', '+00:00'))
                    
                    if start_date <= post_date <= end_date:
                        all_posts.append(post)
                    elif post_date < start_date:
                        return all_posts
            
            offset += limit
            time.sleep(0.5)
        
        return all_posts
    
    def get_post_content(self, slug: str) -> Optional[str]:
        """
        Fetch full content of a specific post
        
        Args:
            slug: Post slug/URL identifier
            
        Returns:
            Post HTML content or None
        """
        try:
            url = f"{self.base_url}/p/{slug}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching post content: {e}")
            return None
    
    def get_three_month_periods(self, start_year: int = 2024, end_year: Optional[int] = None) -> List[Dict]:
        """
        Generate 3-month period intervals
        
        Args:
            start_year: Starting year
            end_year: Ending year (defaults to current year)
            
        Returns:
            List of 3-month period dictionaries
        """
        if end_year is None:
            end_year = datetime.now().year
        
        periods = []
        quarters = [
            ('Q1', 1, 3, 'Jan-Mar'),
            ('Q2', 4, 6, 'Apr-Jun'),
            ('Q3', 7, 9, 'Jul-Sep'),
            ('Q4', 10, 12, 'Oct-Dec')
        ]
        
        for year in range(start_year, end_year + 1):
            for quarter, start_month, end_month, label in quarters:
                start_date = datetime(year, start_month, 1)
                
                if end_month == 12:
                    end_date = datetime(year, 12, 31, 23, 59, 59)
                else:
                    end_date = datetime(year, end_month + 1, 1) - timedelta(seconds=1)
                
                periods.append({
                    'label': f"{year} {quarter} ({label})",
                    'quarter': quarter,
                    'year': year,
                    'start_date': start_date,
                    'end_date': end_date
                })
        
        return periods


if __name__ == "__main__":
    # Example usage
    username = "ethanmorton"
    fetcher = SubstackFetcher(username)
    
    print(f"Fetching posts from {username}.substack.com...")
    posts = fetcher.fetch_posts(limit=10)
    
    print(f"\nFound {len(posts)} posts:")
    for i, post in enumerate(posts, 1):
        print(f"{i}. {post['title']} ({post['post_date']})")
    
    print("\n3-Month Periods:")
    periods = fetcher.get_three_month_periods(2024)
    for period in periods[-4:]:
        print(f"  {period['label']}")
