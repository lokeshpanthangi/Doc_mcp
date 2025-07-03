import requests
import json

class MCPDocumentAnalyzerClient:
    """Client for MCP Document Analyzer API deployed on Render"""
    
    def __init__(self, base_url, api_key):
        """
        Initialize client
        
        Args:
            base_url: Your Render deployment URL (e.g., 'https://your-app.onrender.com')
            api_key: Your OpenAI API key
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def health_check(self):
        """Check if the API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_api_info(self):
        """Get API information and usage instructions"""
        try:
            response = self.session.get(f"{self.base_url}/api-info")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_text(self, content, title="Untitled Document"):
        """Analyze text content"""
        try:
            payload = {
                "api_key": self.api_key,
                "content": content,
                "title": title
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze-text",
                json=payload
            )
            
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_file(self, file_path, auto_analyze=True):
        """Analyze file content (file must be accessible to the server)"""
        try:
            payload = {
                "api_key": self.api_key,
                "file_path": file_path,
                "auto_analyze": auto_analyze
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze-file",
                json=payload
            )
            
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def search_documents(self, query, limit=10):
        """Search documents in the database"""
        try:
            payload = {
                "query": query,
                "limit": limit
            }
            
            response = self.session.post(
                f"{self.base_url}/search-documents",
                json=payload
            )
            
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def add_document(self, title, content, author=None, category=None, tags=None):
        """Add document to the database"""
        try:
            payload = {
                "api_key": self.api_key,
                "title": title,
                "content": content,
                "author": author,
                "category": category,
                "tags": tags
            }
            
            response = self.session.post(
                f"{self.base_url}/add-document",
                json=payload
            )
            
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Replace with your actual Render URL and OpenAI API key
    RENDER_URL = "https://your-app-name.onrender.com"  # Update this!
    OPENAI_API_KEY = "sk-your-openai-api-key-here"     # Update this!
    
    # Initialize client
    client = MCPDocumentAnalyzerClient(RENDER_URL, OPENAI_API_KEY)
    
    # Test health check
    print("=== Health Check ===")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Get API info
    print("\n=== API Info ===")
    info = client.get_api_info()
    print(json.dumps(info, indent=2))
    
    # Test text analysis
    print("\n=== Text Analysis ===")
    text_result = client.analyze_text(
        content="This is an amazing product! I absolutely love using it. The features are incredible and the user experience is fantastic.",
        title="Product Review"
    )
    
    if text_result.get('success'):
        analysis = text_result['data']
        print(f"Sentiment: {analysis['sentiment']['overall_sentiment']}")
        print(f"Confidence: {analysis['sentiment']['confidence']:.2f}")
        print(f"Keywords: {[kw['word'] for kw in analysis['keywords'][:5]]}")
        print(f"Readability Grade: {analysis['readability']['grade_level']}")
    else:
        print(f"Error: {text_result.get('error')}")
    
    # Test adding a document
    print("\n=== Add Document ===")
    add_result = client.add_document(
        title="Sample Document",
        content="This is a sample document for testing the API functionality.",
        author="Test User",
        category="Testing",
        tags=["sample", "test", "api"]
    )
    
    if add_result.get('success'):
        print(f"Document added with ID: {add_result['data'].get('document_id')}")
    else:
        print(f"Error: {add_result.get('error')}")
    
    # Test search
    print("\n=== Search Documents ===")
    search_result = client.search_documents("sample", limit=5)
    
    if search_result.get('success'):
        print(f"Found {search_result['count']} documents")
        for doc in search_result['data']:
            print(f"- {doc['title']} (ID: {doc['id']})")
    else:
        print(f"Error: {search_result.get('error')}")

# Web usage example with JavaScript
web_example = '''
// JavaScript example for web applications
const API_BASE_URL = 'https://your-app-name.onrender.com';
const OPENAI_API_KEY = 'sk-your-openai-api-key-here';

async function analyzeText(content, title) {
    try {
        const response = await fetch(`${API_BASE_URL}/analyze-text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                api_key: OPENAI_API_KEY,
                content: content,
                title: title
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log('Sentiment:', result.data.sentiment.overall_sentiment);
            console.log('Keywords:', result.data.keywords.map(k => k.word));
            return result.data;
        } else {
            console.error('Error:', result.error);
            return null;
        }
    } catch (error) {
        console.error('Request failed:', error);
        return null;
    }
}

// Usage
analyzeText("This is a great product!", "Review");
'''

print("\n=== Web Usage Example ===")
print(web_example)