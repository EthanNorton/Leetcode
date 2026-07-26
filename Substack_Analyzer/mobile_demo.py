#!/usr/bin/env python3
"""
Mobile-Friendly Demo Page
Provides a simplified view optimized for mobile browsers
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Demo data
DEMO_DATA = {
    "mastery_score": 73.4,
    "mastery_level": "Advanced",
    "predictions": [
        {"title": "Exploring Language in AI", "confidence": 0.95, "topics": ["language", "ai", "systems"]},
        {"title": "The Future of AI: Understanding Perspective", "confidence": 0.60, "topics": ["ai", "understanding"]},
        {"title": "Understanding Vision Through Computer", "confidence": 0.50, "topics": ["vision", "computer"]},
    ],
    "books": [
        {"title": "Deep Learning", "author": "Goodfellow et al.", "relevance": 0.92},
        {"title": "Neural Network Methods for NLP", "author": "Yoav Goldberg", "relevance": 0.89},
    ]
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Substack Analyzer - Mobile Demo</title>
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
            margin-bottom: 5px;
        }
        .header p {
            color: #666;
            font-size: 14px;
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
            color: #333;
        }
        .prediction .confidence {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .prediction .topics {
            color: #666;
            font-size: 13px;
            margin-top: 8px;
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
            color: #333;
        }
        .book .author {
            color: #666;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .book .stars {
            color: #ffc107;
            font-size: 18px;
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
        .metric-label {
            color: #666;
            font-size: 14px;
        }
        .metric-value {
            font-weight: bold;
            color: #667eea;
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
        .footer {
            text-align: center;
            color: white;
            padding: 20px;
            font-size: 14px;
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
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            width: 100%;
            margin-top: 10px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Substack Analyzer</h1>
            <p>Mobile Demo - @ethannorton</p>
        </div>

        <div class="card">
            <h2>🎯 Mastery Score</h2>
            <div class="score-circle">
                <div class="number">73.4</div>
                <div class="label">out of 100</div>
            </div>
            <div style="text-align: center;">
                <span class="badge">Advanced Level</span>
            </div>
            <div style="margin-top: 20px;">
                <div class="metric">
                    <span class="metric-label">Consistency</span>
                    <span class="metric-value">100/100</span>
                </div>
                <div class="bar"><div class="bar-fill" style="width: 100%;"></div></div>
                
                <div class="metric">
                    <span class="metric-label">Diversity</span>
                    <span class="metric-value">100/100</span>
                </div>
                <div class="bar"><div class="bar-fill" style="width: 100%;"></div></div>
                
                <div class="metric">
                    <span class="metric-label">Engagement</span>
                    <span class="metric-value">55/100</span>
                </div>
                <div class="bar"><div class="bar-fill" style="width: 55%;"></div></div>
                
                <div class="metric">
                    <span class="metric-label">Depth</span>
                    <span class="metric-value">50/100</span>
                </div>
                <div class="bar"><div class="bar-fill" style="width: 50%;"></div></div>
                
                <div class="metric">
                    <span class="metric-label">Growth</span>
                    <span class="metric-value">50/100</span>
                </div>
                <div class="bar"><div class="bar-fill" style="width: 50%;"></div></div>
            </div>
        </div>

        <div class="card">
            <h2>🔮 Predicted Topics</h2>
            <div class="prediction">
                <h3>Exploring Language in AI</h3>
                <span class="confidence">95% Confidence</span>
                <div class="topics">Topics: language, ai, systems, development</div>
            </div>
            <div class="prediction">
                <h3>The Future of AI: Understanding Perspective</h3>
                <span class="confidence" style="background: #ffc107;">60% Confidence</span>
                <div class="topics">Topics: ai, understanding, applications</div>
            </div>
            <div class="prediction">
                <h3>Understanding Vision Through Computer</h3>
                <span class="confidence" style="background: #fd7e14;">50% Confidence</span>
                <div class="topics">Topics: vision, computer, neural networks</div>
            </div>
        </div>

        <div class="card">
            <h2>📚 Book Recommendations</h2>
            <div class="book">
                <h3>Deep Learning</h3>
                <div class="author">by Ian Goodfellow, Yoshua Bengio, Aaron Courville</div>
                <div class="stars">⭐⭐⭐⭐⭐</div>
                <div style="margin-top: 8px; color: #666; font-size: 13px;">
                    Relevance: 92% - Recommended based on your interest in AI, Machine Learning
                </div>
            </div>
            <div class="book">
                <h3>Neural Network Methods for NLP</h3>
                <div class="author">by Yoav Goldberg</div>
                <div class="stars">⭐⭐⭐⭐⭐</div>
                <div style="margin-top: 8px; color: #666; font-size: 13px;">
                    Relevance: 89% - Recommended based on your interest in NLP, Neural Networks
                </div>
            </div>
        </div>

        <div class="card">
            <h2>💡 Quick Stats</h2>
            <div class="metric">
                <span class="metric-label">Total Articles</span>
                <span class="metric-value">5</span>
            </div>
            <div class="metric">
                <span class="metric-label">Analysis Method</span>
                <span class="metric-value">LSTM + Statistical</span>
            </div>
            <div class="metric">
                <span class="metric-label">Time Decay Rate</span>
                <span class="metric-value">0.95 (Recent Emphasis)</span>
            </div>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Analysis</button>
        </div>

        <div class="footer">
            <p>Built with Python, LSTM, and ❤️</p>
            <p style="margin-top: 10px; font-size: 12px; opacity: 0.9;">
                Full dashboard: http://localhost:8501
            </p>
        </div>
    </div>
</body>
</html>
"""

class MobileDemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML_TEMPLATE.encode())
    
    def log_message(self, format, *args):
        pass  # Suppress logs

if __name__ == "__main__":
    PORT = 8502
    server = HTTPServer(('0.0.0.0', PORT), MobileDemoHandler)
    print(f"\n✅ Mobile Demo Server Running!")
    print(f"📱 Access on your phone: http://localhost:{PORT}")
    print(f"🌐 Or use your server's public IP")
    print(f"\nPress Ctrl+C to stop\n")
    server.serve_forever()
