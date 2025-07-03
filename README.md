# Document Analyzer MCP Server

A comprehensive Model Context Protocol (MCP) server for document analysis with sentiment analysis, keyword extraction, readability scoring, and document management capabilities.

## Features

### Core Analysis Features
- **Sentiment Analysis**: Positive/negative/neutral sentiment with confidence scores
- **Keyword Extraction**: TF-IDF based keyword extraction with relevance scores
- **Readability Scoring**: Multiple readability metrics (Flesch-Kincaid, SMOG, etc.)
- **Basic Statistics**: Word count, sentence count, paragraph count, character count
- **Named Entity Recognition**: Extract people, organizations, locations, etc.
- **Linguistic Features**: POS tagging, dependency parsing, lexical diversity
- **AI-Powered Insights**: Advanced analysis using OpenAI GPT-4

### Document Management
- **Multi-format Support**: PDF, DOCX, HTML, Markdown, Plain Text
- **Document Storage**: SQLite database with metadata
- **Full-text Search**: Fast document search using FTS5
- **Sample Documents**: 15+ pre-loaded sample documents

### MCP Tools
1. `analyze_document(document_id)` - Full document analysis
2. `get_sentiment(text)` - Sentiment for any text
3. `extract_keywords(text, limit)` - Top keywords
4. `add_document(document_data)` - Add new document
5. `search_documents(query)` - Search by content

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for advanced AI insights)

### Setup Steps

1. **Clone/Download the project**
   ```bash
   cd MCP-Docanalysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy English model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Set environment variables**
   ```bash
   # Windows
   set OPENAI_API_KEY=your_openai_api_key_here
   
   # Linux/Mac
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Test the setup**
   ```bash
   python test_setup.py
   ```

## Usage

### Running the MCP Server

```bash
python document_analyzer.py
```

The server will:
1. Initialize the database
2. Load 15+ sample documents
3. Start the MCP server on stdio
4. Wait for MCP client connections

### Using with MCP Clients

The server implements the MCP protocol and can be used with any MCP-compatible client:

1. **Claude Desktop**: Add to your MCP configuration
2. **Custom MCP Clients**: Connect via stdio interface
3. **Development Tools**: Use for testing and development

### Example MCP Client Configuration

```json
{
  "mcpServers": {
    "document-analyzer": {
      "command": "python",
      "args": ["path/to/document_analyzer.py"],
      "env": {
        "OPENAI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## API Reference

### Tool: analyze_document

**Description**: Perform comprehensive analysis of a document

**Parameters**:
- `document_id` (string): ID of the document to analyze

**Returns**: Complete analysis including sentiment, keywords, readability, etc.

**Example**:
```json
{
  "name": "analyze_document",
  "arguments": {
    "document_id": "abc123"
  }
}
```

### Tool: get_sentiment

**Description**: Get sentiment analysis for any text

**Parameters**:
- `text` (string): Text to analyze for sentiment

**Returns**: Sentiment score, label, and confidence

**Example**:
```json
{
  "name": "get_sentiment",
  "arguments": {
    "text": "I love this product! It's amazing."
  }
}
```

### Tool: extract_keywords

**Description**: Extract keywords from text

**Parameters**:
- `text` (string): Text to extract keywords from
- `limit` (integer, optional): Maximum number of keywords (default: 10)

**Returns**: List of keywords with relevance scores

### Tool: add_document

**Description**: Add a new document to the database

**Parameters**:
- `document_data` (object): Document information
  - `title` (string): Document title
  - `content` (string): Document content
  - `file_path` (string, optional): Path to source file
  - `author` (string, optional): Document author
  - `tags` (array, optional): Document tags
  - `category` (string, optional): Document category

**Returns**: Success status and document ID

### Tool: search_documents

**Description**: Search documents by content

**Parameters**:
- `query` (string): Search query
- `limit` (integer, optional): Maximum results (default: 10)

**Returns**: List of matching documents

## Project Structure

```
MCP-Docanalysis/
├── document_analyzer.py    # Main MCP server implementation
├── requirements.txt        # Python dependencies
├── test_setup.py          # Setup verification script
├── README.md              # This file
├── documents.db           # SQLite database (created on first run)
└── sample_documents/      # Sample document files (optional)
```

## Database Schema

### Documents Table
- `id`: Unique document identifier
- `title`: Document title
- `content`: Full document text
- `file_path`: Original file path
- `file_type`: File extension
- `author`: Document author
- `created_date`: Creation timestamp
- `modified_date`: Last modification
- `tags`: JSON array of tags
- `category`: Document category
- `file_size`: File size in bytes
- `content_hash`: SHA256 hash of content

### Analysis Results Table
- `document_id`: Reference to document
- `word_count`: Number of words
- `sentence_count`: Number of sentences
- `paragraph_count`: Number of paragraphs
- `sentiment_score`: Sentiment polarity (-1 to 1)
- `sentiment_label`: positive/negative/neutral
- `keywords`: JSON array of keywords
- `readability_scores`: JSON object with various metrics
- `linguistic_features`: JSON object with NLP features
- `openai_insights`: JSON object with AI analysis

## Deployment

### Local Development

1. Run directly with Python:
   ```bash
   python document_analyzer.py
   ```

2. Use with MCP client applications

### Production Deployment

#### Option 1: Systemd Service (Linux)

1. Create service file `/etc/systemd/system/document-analyzer.service`:
   ```ini
   [Unit]
   Description=Document Analyzer MCP Server
   After=network.target
   
   [Service]
   Type=simple
   User=your_user
   WorkingDirectory=/path/to/MCP-Docanalysis
   Environment=OPENAI_API_KEY=your_api_key
   ExecStart=/usr/bin/python3 document_analyzer.py
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

2. Enable and start:
   ```bash
   sudo systemctl enable document-analyzer
   sudo systemctl start document-analyzer
   ```

#### Option 2: Docker Container

1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   RUN python -m spacy download en_core_web_sm
   
   COPY . .
   
   ENV OPENAI_API_KEY=""
   
   CMD ["python", "document_analyzer.py"]
   ```

2. Build and run:
   ```bash
   docker build -t document-analyzer .
   docker run -e OPENAI_API_KEY=your_key document-analyzer
   ```

#### Option 3: Cloud Deployment

- **AWS Lambda**: Package as serverless function
- **Google Cloud Run**: Deploy as containerized service
- **Azure Container Instances**: Run as managed container
- **Heroku**: Deploy with Procfile

### Using in Other Projects

#### As a Library

```python
from document_analyzer import DocumentAnalyzer, DocumentDatabase

# Initialize components
analyzer = DocumentAnalyzer(openai_api_key="your_key")
db = DocumentDatabase("./documents.db")

# Analyze text
result = await analyzer.analyze_document_content("Your text here", "doc_id")
print(result.sentiment_label)
print(result.keywords)
```

#### As a Microservice

1. Wrap with REST API (Flask/FastAPI)
2. Deploy as containerized service
3. Use HTTP endpoints instead of MCP protocol

#### Integration Examples

- **Content Management Systems**: Analyze uploaded documents
- **Research Tools**: Batch analyze academic papers
- **Business Intelligence**: Analyze customer feedback
- **Educational Platforms**: Assess text complexity

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for AI-powered insights
- `DOC_ANALYZER_DB_PATH`: Custom database path (default: ./documents.db)

### Customization

1. **Add new document parsers**: Extend `DocumentParser` class
2. **Custom analysis features**: Extend `DocumentAnalyzer` class
3. **Additional MCP tools**: Add to `DocumentAnalyzerMCP.setup_tools()`
4. **Database modifications**: Update schema in `DocumentDatabase.init_database()`

## Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **OpenAI API errors**:
   - Check API key validity
   - Verify account has credits
   - Check rate limits

3. **Database locked errors**:
   - Ensure only one server instance running
   - Check file permissions

4. **Import errors**:
   ```bash
   pip install -r requirements.txt
   python test_setup.py
   ```

### Performance Optimization

1. **Database indexing**: Add indexes for frequent queries
2. **Caching**: Implement Redis for analysis results
3. **Async processing**: Use task queues for large documents
4. **Model optimization**: Use smaller spaCy models for speed

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check troubleshooting section
2. Run `python test_setup.py` to verify setup
3. Check logs for error details
4. Create GitHub issue with error details