from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import tempfile
from openai import OpenAI
import traceback

# Add current directory to path to import document_analyzer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_analyzer import DocumentAnalyzer, DocumentDatabase

app = Flask(__name__)
CORS(app)  # Enable CORS for web requests

class SecureDocumentAnalyzer:
    """Document analyzer that requires user's own OpenAI API key"""
    
    def __init__(self, user_api_key):
        if not user_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Create OpenAI client with user's API key
        self.client = OpenAI(api_key=user_api_key)
        
        # Initialize database (shared across all users)
        self.db = DocumentDatabase()
        
        # Create analyzer instance with user's client
        self.analyzer = DocumentAnalyzer()
        # Override the analyzer's OpenAI client with user's
        self.analyzer.client = self.client
    
    def analyze_text(self, content, title="Untitled"):
        """Analyze text content"""
        return self.analyzer.analyze_document_content(content, title)
    
    def analyze_file(self, file_path, auto_analyze=True):
        """Analyze file content"""
        parsed = self.analyzer.parse_document(file_path)
        
        if auto_analyze:
            analysis = self.analyzer.analyze_document_content(
                parsed['content'], 
                parsed['title']
            )
            return {
                "file_info": parsed,
                "analysis": analysis
            }
        else:
            return {"file_info": parsed}
    
    def search_documents(self, query, limit=10):
        """Search documents in database"""
        return self.db.search_documents(query, limit)
    
    def add_document(self, title, content, author=None, category=None, tags=None):
        """Add document to database"""
        document_data = {
            "title": title,
            "content": content,
            "author": author,
            "category": category,
            "tags": tags
        }
        return self.analyzer.handle_add_document({"document_data": document_data})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "MCP Document Analyzer API",
        "version": "1.0.0",
        "message": "Service is running. Provide your OpenAI API key for analysis."
    })

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text content - requires user's OpenAI API key"""
    try:
        data = request.json
        
        # Validate required fields
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400
        
        # Check for API key
        user_api_key = data.get('api_key')
        if not user_api_key:
            return jsonify({
                "success": False,
                "error": "OpenAI API key is required",
                "message": "Get your API key at https://platform.openai.com/api-keys",
                "example": {
                    "api_key": "sk-your-openai-api-key-here",
                    "content": "Your text to analyze",
                    "title": "Document Title"
                }
            }), 400
        
        # Check for content
        content = data.get('content')
        if not content:
            return jsonify({
                "success": False,
                "error": "Content is required"
            }), 400
        
        title = data.get('title', 'Untitled Document')
        
        # Create analyzer with user's API key
        analyzer = SecureDocumentAnalyzer(user_api_key)
        
        # Analyze content
        result = analyzer.analyze_text(content, title)
        
        return jsonify({
            "success": True,
            "data": result,
            "message": "Analysis completed successfully"
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "type": "internal_error"
        }), 500

@app.route('/analyze-file', methods=['POST'])
def analyze_file():
    """Analyze file content - requires user's OpenAI API key"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400
        
        # Check for API key
        user_api_key = data.get('api_key')
        if not user_api_key:
            return jsonify({
                "success": False,
                "error": "OpenAI API key is required",
                "message": "Get your API key at https://platform.openai.com/api-keys"
            }), 400
        
        # Check for file path
        file_path = data.get('file_path')
        if not file_path:
            return jsonify({
                "success": False,
                "error": "File path is required"
            }), 400
        
        auto_analyze = data.get('auto_analyze', True)
        
        # Create analyzer with user's API key
        analyzer = SecureDocumentAnalyzer(user_api_key)
        
        # Analyze file
        result = analyzer.analyze_file(file_path, auto_analyze)
        
        return jsonify({
            "success": True,
            "data": result,
            "message": "File analysis completed successfully"
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except FileNotFoundError:
        return jsonify({
            "success": False,
            "error": "File not found. Please check the file path."
        }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"File analysis failed: {str(e)}",
            "type": "internal_error"
        }), 500

@app.route('/search-documents', methods=['POST'])
def search_documents():
    """Search documents in database"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400
        
        query = data.get('query')
        if not query:
            return jsonify({
                "success": False,
                "error": "Search query is required"
            }), 400
        
        limit = data.get('limit', 10)
        
        # Create database instance (no API key needed for search)
        db = DocumentDatabase()
        results = db.search_documents(query, limit)
        
        return jsonify({
            "success": True,
            "data": results,
            "query": query,
            "count": len(results)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Search failed: {str(e)}"
        }), 500

@app.route('/add-document', methods=['POST'])
def add_document():
    """Add document to database - requires user's OpenAI API key for analysis"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400
        
        # Check for API key
        user_api_key = data.get('api_key')
        if not user_api_key:
            return jsonify({
                "success": False,
                "error": "OpenAI API key is required for document analysis"
            }), 400
        
        # Check required fields
        title = data.get('title')
        content = data.get('content')
        
        if not title or not content:
            return jsonify({
                "success": False,
                "error": "Title and content are required"
            }), 400
        
        author = data.get('author')
        category = data.get('category')
        tags = data.get('tags')
        
        # Create analyzer with user's API key
        analyzer = SecureDocumentAnalyzer(user_api_key)
        
        # Add document
        result = analyzer.add_document(title, content, author, category, tags)
        
        return jsonify({
            "success": True,
            "data": result,
            "message": "Document added successfully"
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to add document: {str(e)}"
        }), 500

@app.route('/api-info', methods=['GET'])
def api_info():
    """Get API information and usage instructions"""
    return jsonify({
        "service": "MCP Document Analyzer API",
        "version": "1.0.0",
        "description": "Secure document analysis API that requires users to provide their own OpenAI API key",
        "endpoints": {
            "/health": "GET - Health check",
            "/analyze-text": "POST - Analyze text content",
            "/analyze-file": "POST - Analyze file content",
            "/search-documents": "POST - Search documents",
            "/add-document": "POST - Add document to database",
            "/api-info": "GET - This information"
        },
        "authentication": {
            "type": "API Key",
            "description": "Provide your OpenAI API key in request body",
            "get_key": "https://platform.openai.com/api-keys"
        },
        "example_request": {
            "url": "/analyze-text",
            "method": "POST",
            "body": {
                "api_key": "sk-your-openai-api-key-here",
                "content": "Your text to analyze",
                "title": "Document Title"
            }
        },
        "supported_file_types": [
            "PDF (.pdf)",
            "Word Documents (.docx, .doc)",
            "Text Files (.txt)",
            "Markdown (.md, .markdown)",
            "HTML (.html, .htm)"
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": [
            "/health",
            "/analyze-text",
            "/analyze-file", 
            "/search-documents",
            "/add-document",
            "/api-info"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "Please try again later"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)