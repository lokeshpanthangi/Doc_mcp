# 🚀 Quick Deploy Commands - Render Deployment

## 📋 **Essential Commands for Render Deployment**

### **1. Prepare Your Project (Run Once)**

```bash
# Navigate to your project
cd "d:\Nani\Assignment - 10\MCP-Docanalysis"

# Run automated setup (Windows)
deploy_setup.bat

# OR run manually:
python deploy_setup.py
```

### **2. Git Commands (If Setup Script Doesn't Work)**

```bash
# Initialize Git (if not done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - MCP Document Analyzer API"

# Add GitHub remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/your-repo-name.git

# Push to GitHub
git push -u origin main
```

### **3. Render Configuration Settings**

**When creating Web Service on Render:**

| Setting | Value |
|---------|-------|
| **Name** | `mcp-document-analyzer` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements_api.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"` |
| **Start Command** | `gunicorn api_server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| **Plan** | `Free` (for testing) or `Starter` (for production) |

### **4. Environment Variables (Add in Render Dashboard)**

```
PYTHON_VERSION=3.11.0
DATABASE_PATH=./documents.db
NLTK_DATA_PATH=./nltk_data
FLASK_ENV=production
```

### **5. Test Your Deployment**

```bash
# Replace 'your-app-name' with your actual Render app name

# Test health endpoint
curl https://your-app-name.onrender.com/health

# Test API info
curl https://your-app-name.onrender.com/api-info

# Test text analysis (with your OpenAI API key)
curl -X POST https://your-app-name.onrender.com/analyze-text \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "sk-your-openai-api-key-here",
    "content": "This is a test document for analysis.",
    "title": "Test Document"
  }'
```

### **6. Update and Redeploy**

```bash
# Make your changes, then:
git add .
git commit -m "Update API functionality"
git push origin main

# Render automatically redeploys! 🎉
```

## 🔧 **Troubleshooting Commands**

### **Fix Common Issues:**

```bash
# If NLTK download fails, try this build command instead:
pip install -r requirements_api.txt && python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Check if all required files exist:
dir api_server.py document_analyzer.py requirements_api.txt Procfile render.yaml

# Verify Git status:
git status

# Check remote repository:
git remote -v
```

### **Emergency Reset:**

```bash
# If Git gets messed up:
rmdir /s .git
git init
git add .
git commit -m "Fresh start"
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main --force
```

## 📱 **Share Your API**

**Your API URL:** `https://your-app-name.onrender.com`

**Share with users:**
1. The API URL above
2. The `client_example.py` file
3. Instructions to get OpenAI API key from: https://platform.openai.com/api-keys

## ⚡ **Super Quick Deploy (Copy-Paste)**

```bash
# 1. Run this in your project directory:
cd "d:\Nani\Assignment - 10\MCP-Docanalysis" && python deploy_setup.py

# 2. Go to render.com, create Web Service, connect GitHub repo

# 3. Use these exact settings:
# Build: pip install -r requirements_api.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
# Start: gunicorn api_server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120

# 4. Add environment variables:
# PYTHON_VERSION=3.11.0
# DATABASE_PATH=./documents.db
# NLTK_DATA_PATH=./nltk_data
# FLASK_ENV=production

# 5. Deploy and test:
# curl https://your-app-name.onrender.com/health
```

## 🎯 **Success Checklist**

- ✅ All files present (api_server.py, requirements_api.txt, etc.)
- ✅ Git repository initialized and pushed to GitHub
- ✅ Render Web Service created and connected to GitHub
- ✅ Build and Start commands configured correctly
- ✅ Environment variables set
- ✅ Deployment successful (green status in Render)
- ✅ Health endpoint returns 200 OK
- ✅ API accepts OpenAI API keys and processes requests

**🎉 You're live! Share your API URL with the world!**