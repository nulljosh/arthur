#!/bin/bash
# One-click deployment to Vercel
echo "📦 Building for production..."
pip freeze > requirements.txt
echo "🚀 Deploying to Vercel..."
vercel --prod
echo "✅ Deployed! Check https://arthur-prod.vercel.app"
