#!/usr/bin/env python3
"""
Dynamic Mobile Substack Analyzer
Search any Substack and get instant AI-powered analysis
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from threading import Thread
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collector import SubstackCollector
from mastery_scorer import MasteryScorer
from article_predictor import ArticlePredictor
from book_recommender import BookRecommender

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Substack Analyzer - Search Any Newsletter</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 10px;
            min-height: 100vh;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
        }
        .header {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            font-size: 24px;
            color: #667eea;
            margin-bottom: 10px;
        }
        .search-box {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .search-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            margin-bottom: 10px;
            transition: border-color 0.3s;
        }
        .search-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .search-btn {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        .search-btn:active {
            transform: scale(0.98);
        }
        .search-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px 20px;
            background: white;
            border-radius: 15px;
            margin-bottom: 15px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results {
            display: none;
        }
        .results.active {
            display: block;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card h2 {
            font-size: 18px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #667eea;
        }
        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: white;
            margin: 20px auto;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        .score-circle .number {
            font-size: 36px;
            font-weight: bold;
        }
        .score-circle .label {
            font-size: 14px;
            opacity: 0.9;
        }
        .badge {
            display: inline-block;
            background: #f3e5f5;
            color: #7b1fa2;
            padding: 6px 12px;
            border-radius: 16px;
            font-weight: bold;
            font-size: 14px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .bar {
            background: #e9ecef;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }
        .bar-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.3s ease;
        }
        .prediction {
            background: #f8f9fa;
            border-left: 4px solid #28a745;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .prediction h3 {
            font-size: 16px;
            margin-bottom: 8px;
        }
        .confidence {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .book {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .book h3 {
            font-size: 16px;
            margin-bottom: 5px;
        }
        .error {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            color: #721c24;
        }
        .examples {
            font-size: 13px;
            color: #666;
            margin-top: 10px;
        }
        .examples a {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }
        .summary-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
        }
        .summary-card h2 {
            color: #2d3748;
        }
        .summary-item {
            background: white;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .summary-label {
            font-weight: 600;
            color: #4a5568;
        }
        .summary-value {
            font-weight: bold;
            color: #667eea;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Substack Analyzer</h1>
            <p>Search any Substack newsletter for instant AI-powered analysis</p>
        </div>

        <div class="search-box">
            <input 
                type="text" 
                id="substackUrl" 
                class="search-input" 
                placeholder="Enter Substack URL (e.g., example.substack.com)"
                value="https://substack.com/@ethannorton"
            >
            <button class="search-btn" onclick="analyzeSubstack()">
                🔍 Analyze Newsletter
            </button>
            <div class="examples">
                💡 Examples: 
                <a href="#" onclick="setUrl('platformer.news'); return false;">platformer.news</a> | 
                <a href="#" onclick="setUrl('stratechery.com'); return false;">stratechery.com</a>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <h3 style="color: #667eea; margin-bottom: 10px;">Analyzing Newsletter...</h3>
            <p style="color: #666;">Fetching articles, training LSTM, calculating scores...</p>
            <p style="color: #999; font-size: 13px; margin-top: 10px;">This may take 15-30 seconds</p>
        </div>

        <div class="results" id="results">
            <!-- Results will be inserted here dynamically -->
        </div>
    </div>

    <script>
        function setUrl(url) {
            document.getElementById('substackUrl').value = 'https://' + url;
        }

        async function analyzeSubstack() {
            const url = document.getElementById('substackUrl').value.trim();
            
            if (!url) {
                alert('Please enter a Substack URL');
                return;
            }

            // Show loading
            document.getElementById('loading').classList.add('active');
            document.getElementById('results').classList.remove('active');
            document.getElementById('results').innerHTML = '';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: url })
                });

                const data = await response.json();

                // Hide loading
                document.getElementById('loading').classList.remove('active');

                if (data.error) {
                    displayError(data.error);
                } else {
                    displayResults(data);
                }
            } catch (error) {
                document.getElementById('loading').classList.remove('active');
                displayError('Failed to analyze Substack. Please check the URL and try again.');
            }
        }

        function displayError(message) {
            document.getElementById('results').innerHTML = `
                <div class="card">
                    <div class="error">
                        <strong>❌ Error:</strong> ${message}
                    </div>
                    <p style="margin-top: 15px; color: #666;">
                        <strong>Tips:</strong><br>
                        • Make sure the URL is correct<br>
                        • Try the full URL (e.g., https://example.substack.com)<br>
                        • Some Substacks may not have public RSS feeds
                    </p>
                </div>
            `;
            document.getElementById('results').classList.add('active');
        }

        function displayResults(data) {
            const resultsHtml = `
                <div class="summary-card">
                    <h2>📊 Quick Summary</h2>
                    <div class="summary-item">
                        <span class="summary-label">Articles Analyzed</span>
                        <span class="summary-value">${data.total_articles}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Mastery Level</span>
                        <span class="summary-value">${data.mastery_level}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Top Strength</span>
                        <span class="summary-value" style="font-size: 14px;">${data.top_strength}</span>
                    </div>
                </div>

                <div class="card">
                    <h2>🎯 Mastery Score</h2>
                    <div class="score-circle">
                        <div class="number">${data.mastery_score}</div>
                        <div class="label">out of 100</div>
                    </div>
                    <div style="text-align: center; margin-bottom: 20px;">
                        <span class="badge">${data.mastery_level}</span>
                    </div>
                    ${generateMetrics(data.metrics)}
                </div>

                <div class="card">
                    <h2>🔮 Predicted Topics</h2>
                    <p style="color: #666; margin-bottom: 15px; font-size: 14px;">
                        AI-generated suggestions for future articles
                    </p>
                    ${generatePredictions(data.predictions)}
                </div>

                <div class="card">
                    <h2>📚 Book Recommendations</h2>
                    <p style="color: #666; margin-bottom: 15px; font-size: 14px;">
                        Based on content themes and predicted topics
                    </p>
                    ${generateBooks(data.books)}
                </div>

                <div class="card">
                    <h2>💡 Key Insights</h2>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <strong style="color: #28a745;">💪 Strengths:</strong><br>
                        ${data.strengths.map(s => `• ${s}`).join('<br>')}
                    </div>
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <strong style="color: #856404;">🎯 Growth Areas:</strong><br>
                        ${data.growth_areas.map(g => `• ${g}`).join('<br>')}
                    </div>
                </div>

                <div class="card" style="text-align: center;">
                    <button class="search-btn" onclick="document.getElementById('substackUrl').scrollIntoView({behavior: 'smooth'})">
                        🔍 Analyze Another Newsletter
                    </button>
                </div>
            `;

            document.getElementById('results').innerHTML = resultsHtml;
            document.getElementById('results').classList.add('active');
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }

        function generateMetrics(metrics) {
            return Object.entries(metrics).map(([key, value]) => {
                const label = key.replace('_score', '').replace('_', ' ');
                const capitalizedLabel = label.charAt(0).toUpperCase() + label.slice(1);
                return `
                    <div class="metric">
                        <span>${capitalizedLabel}</span>
                        <span style="font-weight: bold; color: #667eea;">${value}/100</span>
                    </div>
                    <div class="bar"><div class="bar-fill" style="width: ${value}%;"></div></div>
                `;
            }).join('');
        }

        function generatePredictions(predictions) {
            return predictions.slice(0, 5).map((pred, i) => {
                const confColor = pred.confidence > 0.75 ? '#28a745' : pred.confidence > 0.5 ? '#ffc107' : '#fd7e14';
                return `
                    <div class="prediction">
                        <h3>${pred.title}</h3>
                        <span class="confidence" style="background: ${confColor};">
                            ${(pred.confidence * 100).toFixed(0)}% Confidence
                        </span>
                        <div style="margin-top: 8px; color: #666; font-size: 13px;">
                            Topics: ${pred.topics.slice(0, 5).join(', ')}
                        </div>
                    </div>
                `;
            }).join('');
        }

        function generateBooks(books) {
            return books.slice(0, 5).map(book => {
                const stars = '⭐'.repeat(Math.round(book.relevance_score * 5));
                return `
                    <div class="book">
                        <h3>${book.title}</h3>
                        <div style="color: #666; font-size: 14px; margin-bottom: 5px;">
                            by ${book.author}
                        </div>
                        <div style="font-size: 18px;">${stars}</div>
                        <div style="color: #666; font-size: 13px; margin-top: 8px;">
                            ${book.reason}
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Load on page load if URL is set
        window.addEventListener('load', function() {
            const urlParam = new URLSearchParams(window.location.search).get('url');
            if (urlParam) {
                document.getElementById('substackUrl').value = urlParam;
                analyzeSubstack();
            }
        });
    </script>
</body>
</html>
"""

class DynamicAnalyzerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            substack_url = data.get('url', '')
            
            try:
                # Perform analysis
                result = analyze_substack_url(substack_url)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                error_result = {'error': str(e)}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_result).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logs


def analyze_substack_url(url):
    """Analyze a Substack URL and return results"""
    print(f"\n🔍 Analyzing: {url}")
    
    # Collect articles
    collector = SubstackCollector(url)
    articles_df = collector.collect_articles(max_articles=20)
    
    if articles_df.empty:
        raise Exception("No articles found. Please check the Substack URL.")
    
    print(f"✓ Collected {len(articles_df)} articles")
    
    # Calculate mastery
    scorer = MasteryScorer()
    mastery_results = scorer.analyze_mastery(articles_df, articles_df)
    print(f"✓ Mastery score: {mastery_results['overall_score']:.1f}")
    
    # Generate predictions
    predictor = ArticlePredictor()
    predictor.fit(articles_df)
    predictions = predictor.predict_next_articles(num_predictions=8)
    print(f"✓ Generated {len(predictions)} predictions")
    
    # Get book recommendations
    recommender = BookRecommender()
    books = recommender.recommend_books(articles_df, predictions, num_books=8)
    print(f"✓ {len(books)} books recommended")
    
    # Format response
    return {
        'total_articles': len(articles_df),
        'mastery_score': round(mastery_results['overall_score'], 1),
        'mastery_level': mastery_results['mastery_level'],
        'top_strength': mastery_results['strengths'][0] if mastery_results['strengths'] else 'Growing',
        'metrics': {k: round(v, 1) for k, v in mastery_results['metrics'].items()},
        'predictions': predictions,
        'books': books,
        'strengths': mastery_results['strengths'][:3],
        'growth_areas': mastery_results['growth_areas'][:3],
    }


if __name__ == "__main__":
    PORT = 8503
    server = HTTPServer(('0.0.0.0', PORT), DynamicAnalyzerHandler)
    print(f"\n{'='*70}")
    print(f"✅ DYNAMIC SUBSTACK ANALYZER RUNNING!")
    print(f"{'='*70}")
    print(f"\n📱 Mobile Access: http://localhost:{PORT}")
    print(f"🌐 Network Access: http://52.2.175.83:{PORT}")
    print(f"\n🔍 Features:")
    print(f"  • Search ANY Substack newsletter")
    print(f"  • Real-time AI analysis")
    print(f"  • LSTM predictions with time-decay")
    print(f"  • Instant mastery scoring")
    print(f"  • Book recommendations")
    print(f"\n💡 Just enter a Substack URL and click Analyze!")
    print(f"\nPress Ctrl+C to stop\n")
    print(f"{'='*70}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        server.shutdown()
