# 🚀 Render Deployment Guide - MCP Document Analyzer

## 📋 **Prerequisites**

- ✅ GitHub account
- ✅ Render account (free tier available)
- ✅ Your project files ready
- ✅ Git installed on your computer

## 📁 **Required Files for Deployment**

Make sure you have these files in your project:

```
MCP-Docanalysis/
├── api_server.py          # Main Flask API server
├── document_analyzer.py   # Your existing MCP code
├── requirements_api.txt   # Python dependencies
├── Procfile              # Render process configuration
├── render.yaml           # Render deployment config
├── client_example.py     # Usage examples
├── .env                  # Environment variables (local only)
└── documents.db          # Database file
```

## 🔧 **Step 1: Prepare Your Repository**

### **1.1 Initialize Git Repository (if not already done)**

```bash
# Navigate to your project directory
cd "d:\Nani\Assignment - 10\MCP-Docanalysis"

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit - MCP Document Analyzer API"
```

### **1.2 Create .gitignore File**

```bash
# Create .gitignore to exclude sensitive files
echo "# Environment variables" > .gitignore
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "*.pyo" >> .gitignore
echo "*.pyd" >> .gitignore
echo ".Python" >> .gitignore
echo "env/" >> .gitignore
echo "venv/" >> .gitignore
echo ".venv/" >> .gitignore
echo "pip-log.txt" >> .gitignore
echo "pip-delete-this-directory.txt" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "Thumbs.db" >> .gitignore

# Add and commit .gitignore
git add .gitignore
git commit -m "Add .gitignore file"
```

### **1.3 Push to GitHub**

```bash
# Create a new repository on GitHub first, then:
# Replace 'yourusername' and 'your-repo-name' with actual values

git remote add origin https://github.com/yourusername/your-repo-name.git
git branch -M main
git push -u origin main
```

## 🌐 **Step 2: Deploy to Render**

### **2.1 Create Render Account**

1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account
3. Authorize Render to access your repositories

### **2.2 Create New Web Service**

1. **Click "New +"** in Render dashboard
2. **Select "Web Service"**
3. **Connect your GitHub repository**
   - Choose your MCP Document Analyzer repository
   - Click "Connect"

### **2.3 Configure Deployment Settings**

**Basic Settings:**
- **Name**: `mcp-document-analyzer` (or your preferred name)
- **Region**: Choose closest to your users
- **Branch**: `main`
- **Root Directory**: Leave empty (unless your code is in a subdirectory)

**Build & Deploy:**
- **Runtime**: `Python 3`
- **Build Command**: 
  ```bash
  pip install -r requirements_api.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
  ```
- **Start Command**: 
  ```bash
  gunicorn api_server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
  ```

**Plan:**
- **Select "Free"** (for testing) or **"Starter"** (for production)

### **2.4 Environment Variables**

In the **Environment Variables** section, add:

```
PYTHON_VERSION=3.11.0
DATABASE_PATH=./documents.db
NLTK_DATA_PATH=./nltk_data
FLASK_ENV=production
```

### **2.5 Deploy**

1. **Click "Create Web Service"**
2. **Wait for deployment** (usually 5-10 minutes)
3. **Monitor the build logs** for any errors

## 📝 **Step 3: Verify Deployment**

### **3.1 Test Health Endpoint**

Once deployed, your service will be available at:
```
https://your-app-name.onrender.com
```

Test the health endpoint:
```bash
curl https://your-app-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "MCP Document Analyzer API",
  "version": "1.0.0",
  "message": "Service is running. Provide your OpenAI API key for analysis."
}
```

### **3.2 Test API Info Endpoint**

```bash
curl https://your-app-name.onrender.com/api-info
```

### **3.3 Test Text Analysis**

```bash
curl -X POST https://your-app-name.onrender.com/analyze-text \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "sk-your-openai-api-key-here",
    "content": "This is a test document for analysis.",
    "title": "Test Document"
  }'
```

## 🔧 **Step 4: Update and Redeploy**

### **4.1 Make Changes Locally**

```bash
# Make your changes to the code
# Then commit and push

git add .
git commit -m "Update API functionality"
git push origin main
```

### **4.2 Automatic Redeployment**

Render automatically redeploys when you push to the connected branch!

### **4.3 Manual Redeploy**

If needed, you can manually redeploy:
1. Go to your service in Render dashboard
2. Click **"Manual Deploy"**
3. Select **"Deploy latest commit"**

## 🛠️ **Troubleshooting**

### **Common Issues and Solutions**

#### **1. Build Fails - Dependencies**
```bash
# Check requirements_api.txt has all dependencies
# Common missing dependencies:
echo "gunicorn==21.2.0" >> requirements_api.txt
echo "Flask-CORS==4.0.0" >> requirements_api.txt
```

#### **2. NLTK Data Download Fails**
```bash
# Update build command to handle NLTK downloads better:
pip install -r requirements_api.txt && python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

#### **3. Database Issues**
```bash
# Ensure database is created on startup
# Add to your api_server.py:
# Create database tables on startup
with app.app_context():
    db = DocumentDatabase()
```

#### **4. Port Issues**
```bash
# Ensure your app uses the PORT environment variable
# In api_server.py:
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
```

### **Check Deployment Logs**

1. Go to your service in Render dashboard
2. Click **"Logs"** tab
3. Look for error messages

### **Common Error Messages**

- **"Module not found"**: Add missing dependency to requirements_api.txt
- **"Port already in use"**: Ensure you're using `$PORT` environment variable
- **"NLTK data not found"**: Check NLTK download in build command
- **"Database locked"**: Ensure proper database initialization

## 📱 **Step 5: Share Your API**

### **5.1 API Documentation**

Share this information with users:

**Base URL**: `https://your-app-name.onrender.com`

**Required**: Users must provide their own OpenAI API key

**Get API Key**: https://platform.openai.com/api-keys

### **5.2 Usage Examples**

Provide users with the `client_example.py` file:

```python
# Update the URL in client_example.py
RENDER_URL = "https://your-actual-app-name.onrender.com"
```

### **5.3 API Endpoints**

- `GET /health` - Health check
- `GET /api-info` - API information
- `POST /analyze-text` - Analyze text content
- `POST /analyze-file` - Analyze file content
- `POST /search-documents` - Search documents
- `POST /add-document` - Add document to database

## 🔒 **Security Best Practices**

### **5.1 API Key Protection**

- ✅ Users provide their own OpenAI API keys
- ✅ API keys are not stored on your server
- ✅ Each request uses the user's own credits

### **5.2 Rate Limiting (Optional)**

Add rate limiting to prevent abuse:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
```

### **5.3 Input Validation**

- ✅ Validate all input parameters
- ✅ Sanitize file paths
- ✅ Limit request sizes

## 💰 **Cost Considerations**

### **Render Costs**
- **Free Tier**: 750 hours/month (sleeps after 15 min inactivity)
- **Starter Plan**: $7/month (always on)

### **OpenAI Costs**
- **Users pay their own OpenAI costs**
- **You don't pay for user API usage**
- **Your server only handles processing**

## 🎯 **Quick Commands Summary**

```bash
# 1. Prepare repository
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/repo-name.git
git push -u origin main

# 2. Deploy to Render
# - Go to render.com
# - Connect GitHub repo
# - Configure settings
# - Deploy

# 3. Test deployment
curl https://your-app-name.onrender.com/health

# 4. Update and redeploy
git add .
git commit -m "Updates"
git push origin main
# Render auto-deploys!
```

## 🎉 **Success!**

Your MCP Document Analyzer is now deployed and ready for users! They can:

✅ **Use their own OpenAI API keys** (no cost to you)
✅ **Analyze text and documents** via your API
✅ **Search and manage documents** in the shared database
✅ **Access from anywhere** via the web API

**Your Render URL**: `https://your-app-name.onrender.com`

Share this URL and the `client_example.py` file with anyone who wants to use your MCP Document Analyzer! 🚀