#!/bin/bash
# Run Substack Dashboard
# Quick start script for the Substack Content Dashboard

echo "🚀 Starting Substack Content Dashboard..."
echo ""
echo "Installing dependencies..."
pip install -q -r requirements_substack.txt

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "🌐 Launching dashboard..."
echo "   The dashboard will open in your browser"
echo "   Press Ctrl+C to stop the server"
echo ""

streamlit run substack_dashboard.py
