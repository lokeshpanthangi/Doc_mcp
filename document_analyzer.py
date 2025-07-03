#!/usr/bin/env python3
"""
Document Analyzer MCP Server
A comprehensive MCP server for document analysis with sentiment, keywords, and readability scoring.
"""

import os
import json
import sqlite3
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# External libraries
import openai
import nltk
import spacy
from textblob import TextBlob
import PyPDF2
import docx
from bs4 import BeautifulSoup
import markdown
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# FastMCP imports
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    title: str
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    tags: List[str] = None
    category: Optional[str] = None
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None

@dataclass
class AnalysisResult:
    """Analysis result structure"""
    document_id: str
    word_count: int
    sentence_count: int
    paragraph_count: int
    char_count: int
    sentiment_score: float
    sentiment_label: str
    sentiment_confidence: float
    emotions: Dict[str, float]
    keywords: List[Dict[str, Union[str, float]]]
    named_entities: List[Dict[str, str]]
    readability_scores: Dict[str, float]
    linguistic_features: Dict[str, Any]
    openai_insights: Dict[str, Any]
    analysis_timestamp: datetime

class DocumentParser:
    """Handles parsing of different document types"""
    
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Parse PDF file"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error parsing PDF: {str(e)}")
    
    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Parse DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error parsing DOCX: {str(e)}")
    
    @staticmethod
    def parse_html(file_path: str) -> str:
        """Parse HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                return soup.get_text().strip()
        except Exception as e:
            raise Exception(f"Error parsing HTML: {str(e)}")
    
    @staticmethod
    def parse_markdown(file_path: str) -> str:
        """Parse Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text().strip()
        except Exception as e:
            raise Exception(f"Error parsing Markdown: {str(e)}")
    
    @staticmethod
    def parse_text(file_path: str) -> str:
        """Parse plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            raise Exception(f"Error parsing text file: {str(e)}")
    
    @classmethod
    def parse_document(cls, file_path: str) -> str:
        """Parse document based on file extension"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        parsers = {
            '.pdf': cls.parse_pdf,
            '.docx': cls.parse_docx,
            '.doc': cls.parse_docx,
            '.html': cls.parse_html,
            '.htm': cls.parse_html,
            '.md': cls.parse_markdown,
            '.markdown': cls.parse_markdown,
            '.txt': cls.parse_text,
        }
        
        if extension not in parsers:
            raise Exception(f"Unsupported file type: {extension}")
        
        return parsers[extension](str(file_path))

class DocumentAnalyzer:
    """Main document analysis engine"""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.setup_nltk()
        self.setup_spacy()
    
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('chunkers/maxent_ne_chunker')
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')
    
    def setup_spacy(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
            raise
    
    def get_basic_stats(self, text: str) -> Dict[str, int]:
        """Get basic text statistics"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        paragraphs = text.split('\n\n')
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'char_count': len(text)
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[float, str]]:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            'sentiment_score': polarity,
            'sentiment_label': label,
            'sentiment_confidence': abs(polarity),
            'subjectivity': subjectivity
        }
    
    def extract_keywords(self, text: str, limit: int = 10) -> List[Dict[str, Union[str, float]]]:
        """Extract keywords using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [
                {'keyword': keyword, 'score': float(score)}
                for keyword, score in keyword_scores[:limit]
            ]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_named_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using spaCy"""
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_)
                })
            return entities
        except Exception as e:
            logger.error(f"Error extracting named entities: {str(e)}")
            return []
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate various readability scores"""
        try:
            return {
                'flesch_kincaid_grade': textstat.flesch_kincaid().grade(text),
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'smog_index': textstat.smog_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'gunning_fog': textstat.gunning_fog(text),
                'reading_time_minutes': textstat.reading_time(text, ms_per_char=14.69)
            }
        except Exception as e:
            logger.error(f"Error calculating readability: {str(e)}")
            return {}
    
    def analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features using spaCy"""
        try:
            doc = self.nlp(text)
            
            pos_counts = {}
            dep_counts = {}
            
            for token in doc:
                if not token.is_space:
                    pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
                    dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1
            
            return {
                'pos_distribution': pos_counts,
                'dependency_distribution': dep_counts,
                'avg_sentence_length': len([token for token in doc if not token.is_space]) / len(list(doc.sents)),
                'lexical_diversity': len(set([token.lemma_.lower() for token in doc if token.is_alpha])) / len([token for token in doc if token.is_alpha])
            }
        except Exception as e:
            logger.error(f"Error analyzing linguistic features: {str(e)}")
            return {}
    
    async def get_openai_insights(self, text: str) -> Dict[str, Any]:
        """Get advanced insights from OpenAI GPT-4"""
        try:
            prompt = f"""
            Analyze the following text and provide insights in JSON format:
            
            1. Detailed sentiment analysis with emotions (joy, anger, fear, sadness, surprise, disgust)
            2. Main themes and topics
            3. Writing style assessment
            4. Tone and mood
            5. Key insights and summary
            
            Text: {text[:4000]}  # Limit text length for API
            
            Respond only with valid JSON.
            """
            
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error getting OpenAI insights: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def analyze_document_content(self, text: str, document_id: str) -> AnalysisResult:
        """Perform comprehensive document analysis"""
        try:
            # Basic statistics
            stats = self.get_basic_stats(text)
            
            # Sentiment analysis
            sentiment = self.analyze_sentiment(text)
            
            # Keywords extraction
            keywords = self.extract_keywords(text)
            
            # Named entities
            entities = self.extract_named_entities(text)
            
            # Readability scores
            readability = self.calculate_readability(text)
            
            # Linguistic features
            linguistic = self.analyze_linguistic_features(text)
            
            # OpenAI insights
            openai_insights = await self.get_openai_insights(text)
            
            # Extract emotions from OpenAI response
            emotions = openai_insights.get('emotions', {})
            
            return AnalysisResult(
                document_id=document_id,
                word_count=stats['word_count'],
                sentence_count=stats['sentence_count'],
                paragraph_count=stats['paragraph_count'],
                char_count=stats['char_count'],
                sentiment_score=sentiment['sentiment_score'],
                sentiment_label=sentiment['sentiment_label'],
                sentiment_confidence=sentiment['sentiment_confidence'],
                emotions=emotions,
                keywords=keywords,
                named_entities=entities,
                readability_scores=readability,
                linguistic_features=linguistic,
                openai_insights=openai_insights,
                analysis_timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            raise

class DocumentDatabase:
    """Database manager for documents and analysis results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                file_path TEXT,
                file_type TEXT,
                author TEXT,
                created_date TEXT,
                modified_date TEXT,
                tags TEXT,
                category TEXT,
                file_size INTEGER,
                content_hash TEXT,
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                word_count INTEGER,
                sentence_count INTEGER,
                paragraph_count INTEGER,
                char_count INTEGER,
                sentiment_score REAL,
                sentiment_label TEXT,
                sentiment_confidence REAL,
                emotions TEXT,
                keywords TEXT,
                named_entities TEXT,
                readability_scores TEXT,
                linguistic_features TEXT,
                openai_insights TEXT,
                analysis_timestamp TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Create FTS5 virtual table for full-text search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                document_id,
                title,
                content,
                author,
                tags,
                category
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_document(self, document_id: str, title: str, content: str, metadata: DocumentMetadata) -> bool:
        """Add document to database"""
        try:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, title, content, file_path, file_type, author, created_date, 
                 modified_date, tags, category, file_size, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_id, title, content, metadata.file_path, metadata.file_type,
                metadata.author, metadata.created_date.isoformat() if metadata.created_date else None,
                metadata.modified_date.isoformat() if metadata.modified_date else None,
                json.dumps(metadata.tags or []), metadata.category, metadata.file_size, content_hash
            ))
            
            # Add to FTS table
            cursor.execute('''
                INSERT OR REPLACE INTO documents_fts 
                (document_id, title, content, author, tags, category)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                document_id, title, content, metadata.author or '',
                ' '.join(metadata.tags or []), metadata.category or ''
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM documents WHERE id = ?', (document_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                doc = dict(zip(columns, row))
                doc['tags'] = json.loads(doc['tags']) if doc['tags'] else []
                conn.close()
                return doc
            
            conn.close()
            return None
        except Exception as e:
            logger.error(f"Error getting document: {str(e)}")
            return None
    
    def save_analysis_result(self, result: AnalysisResult) -> bool:
        """Save analysis result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_results 
                (document_id, word_count, sentence_count, paragraph_count, char_count,
                 sentiment_score, sentiment_label, sentiment_confidence, emotions,
                 keywords, named_entities, readability_scores, linguistic_features,
                 openai_insights, analysis_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.document_id, result.word_count, result.sentence_count,
                result.paragraph_count, result.char_count, result.sentiment_score,
                result.sentiment_label, result.sentiment_confidence,
                json.dumps(result.emotions), json.dumps(result.keywords),
                json.dumps(result.named_entities), json.dumps(result.readability_scores),
                json.dumps(result.linguistic_features), json.dumps(result.openai_insights),
                result.analysis_timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving analysis result: {str(e)}")
            return False
    
    def get_analysis_result(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result for document"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM analysis_results 
                WHERE document_id = ? 
                ORDER BY analysis_timestamp DESC 
                LIMIT 1
            ''', (document_id,))
            
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                result = dict(zip(columns, row))
                
                # Parse JSON fields
                json_fields = ['emotions', 'keywords', 'named_entities', 'readability_scores', 
                              'linguistic_features', 'openai_insights']
                for field in json_fields:
                    if result[field]:
                        result[field] = json.loads(result[field])
                
                conn.close()
                return result
            
            conn.close()
            return None
        except Exception as e:
            logger.error(f"Error getting analysis result: {str(e)}")
            return None
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents using full-text search"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.*, rank FROM documents d
                JOIN documents_fts fts ON d.id = fts.document_id
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (query, limit))
            
            results = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                doc = dict(zip(columns, row))
                doc['tags'] = json.loads(doc['tags']) if doc['tags'] else []
                results.append(doc)
            
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

# Initialize FastMCP server
app = FastMCP("document-analyzer")

# Global variables for database and analyzer
db = None
analyzer = None
parser = None

# Components will be initialized by the function below
    
@app.tool()
async def analyze_document(document_id: str) -> str:
    """Perform comprehensive analysis of a document"""
    # Get document from database
    doc = db.get_document(document_id)
    if not doc:
        raise Exception(f"Document with ID {document_id} not found")
    
    # Check if analysis already exists and is recent
    existing_analysis = db.get_analysis_result(document_id)
    if existing_analysis:
        # Return cached result
        return json.dumps(existing_analysis, indent=2)
    
    # Perform analysis
    analysis_result = await analyzer.analyze_document_content(doc['content'], document_id)
    
    # Save analysis result
    db.save_analysis_result(analysis_result)
    
    # Return result
    result_dict = asdict(analysis_result)
    result_dict['analysis_timestamp'] = result_dict['analysis_timestamp'].isoformat()
    
    return json.dumps(result_dict, indent=2)

@app.tool()
async def get_sentiment(text: str) -> str:
    """Get sentiment analysis for any text"""
    sentiment = analyzer.analyze_sentiment(text)
    return json.dumps(sentiment, indent=2)

@app.tool()
async def extract_keywords(text: str, limit: int = 10) -> str:
    """Extract keywords from text"""
    keywords = analyzer.extract_keywords(text, limit)
    return json.dumps(keywords, indent=2)
    
@app.tool()
async def add_document(document_data: dict) -> str:
    """Add a new document to the database with text content"""
    # Generate document ID
    document_id = hashlib.sha256(
        f"{document_data['title']}{document_data['content']}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]
    
    # Create metadata
    metadata = DocumentMetadata(
        title=document_data["title"],
        author=document_data.get("author"),
        tags=document_data.get("tags", []),
        category=document_data.get("category"),
        created_date=datetime.now(),
        modified_date=datetime.now()
    )
    
    # Use provided content directly
    content = document_data["content"]
    
    # Add document to database
    success = db.add_document(document_id, document_data["title"], content, metadata)
    
    if success:
        return json.dumps({
            "success": True,
            "document_id": document_id,
            "message": "Document added successfully",
            "word_count": len(content.split()),
            "char_count": len(content)
        }, indent=2)
    else:
        raise Exception("Failed to add document to database")

@app.tool()
async def search_documents(query: str, limit: int = 10) -> str:
    """Search documents by content"""
    results = db.search_documents(query, limit)
    return json.dumps(results, indent=2)
    
@app.tool()
async def add_document_from_file(file_path: str, title: str = None, author: str = None, tags: list = None, category: str = None, auto_analyze: bool = False) -> str:
    """Add a new document from file upload"""
    # Validate file exists
    if not os.path.exists(file_path):
        raise Exception(f"File not found: {file_path}")
    
    # Get file information
    file_path_obj = Path(file_path)
    file_extension = file_path_obj.suffix.lower()
    file_size = os.path.getsize(file_path)
    
    # Validate file type
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.html', '.htm', '.md', '.markdown']
    if file_extension not in supported_extensions:
        raise Exception(f"Unsupported file type: {file_extension}. Supported types: {', '.join(supported_extensions)}")
    
    # Parse document content
    try:
        content = parser.parse_document(file_path)
        if not content.strip():
            raise Exception("No text content could be extracted from the file")
    except Exception as e:
        raise Exception(f"Error parsing file: {str(e)}")
    
    # Use custom title or filename
    if not title:
        title = file_path_obj.stem
    
    # Generate document ID
    document_id = hashlib.sha256(
        f"{title}{content[:1000]}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]
    
    # Create metadata
    metadata = DocumentMetadata(
        title=title,
        author=author,
        tags=tags or [],
        category=category,
        file_path=file_path,
        file_type=file_extension,
        file_size=file_size,
        created_date=datetime.now(),
        modified_date=datetime.now()
    )
    
    # Add document to database
    success = db.add_document(document_id, title, content, metadata)
    
    if not success:
        raise Exception("Failed to add document to database")
    
    # Prepare response
    response_data = {
        "success": True,
        "document_id": document_id,
        "message": "Document uploaded and added successfully",
        "file_info": {
            "filename": file_path_obj.name,
            "file_type": file_extension,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        },
        "content_stats": {
            "word_count": len(content.split()),
            "char_count": len(content),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()])
        }
    }
    
    # Auto-analyze if requested
    if auto_analyze:
        try:
            analysis_result = await analyzer.analyze_document_content(content, document_id)
            db.save_analysis_result(analysis_result)
            
            # Add analysis summary to response
            response_data["analysis_summary"] = {
                "sentiment_label": analysis_result.sentiment_label,
                "sentiment_score": analysis_result.sentiment_score,
                "top_keywords": [kw["keyword"] for kw in analysis_result.keywords[:5]],
                "readability_grade": analysis_result.readability_scores.get("flesch_kincaid_grade", "N/A")
            }
            response_data["message"] += " and analyzed"
        except Exception as e:
            logger.warning(f"Auto-analysis failed: {str(e)}")
            response_data["analysis_warning"] = f"Document added but analysis failed: {str(e)}"
    
    return json.dumps(response_data, indent=2)

def initialize_components():
    """Initialize global components"""
    global db, analyzer, parser
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    database_path = os.getenv("DOC_ANALYZER_DB_PATH", "./documents.db")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise Exception("OPENAI_API_KEY environment variable is required")
    
    # Initialize components
    db = DocumentDatabase(database_path)
    analyzer = DocumentAnalyzer(openai_api_key)
    parser = DocumentParser()
    
    logger.info("Components initialized successfully")

async def add_sample_documents():
    """Add sample documents to the database"""
    # We need to add documents directly to the database instead of using the tool
    # since the tool is not callable directly
    logger.info("Adding sample documents to database...")
    
    sample_documents = [
        {
            "title": "Climate Change Impact Report",
            "content": """Climate change represents one of the most pressing challenges of our time. The scientific consensus is clear: human activities are the primary driver of recent climate change. Rising global temperatures have led to melting ice caps, rising sea levels, and more frequent extreme weather events. The impacts are far-reaching, affecting ecosystems, agriculture, water resources, and human health. Immediate action is required to mitigate these effects through renewable energy adoption, carbon emission reductions, and sustainable practices. The window for effective action is narrowing, making urgent policy changes and international cooperation essential.""",
            "author": "Dr. Sarah Johnson",
            "category": "Environmental Science",
            "tags": ["climate", "environment", "sustainability", "global warming"]
        },
        {
            "title": "Artificial Intelligence in Healthcare",
            "content": """Artificial intelligence is revolutionizing healthcare delivery and patient outcomes. Machine learning algorithms can now diagnose diseases with accuracy comparable to human specialists. AI-powered medical imaging systems can detect cancer, fractures, and other conditions earlier than traditional methods. Natural language processing helps analyze patient records and clinical notes, extracting valuable insights for treatment decisions. However, challenges remain including data privacy, algorithm bias, and the need for regulatory frameworks. The integration of AI in healthcare promises to improve efficiency, reduce costs, and ultimately save lives through more precise and personalized medicine.""",
            "author": "Dr. Michael Chen",
            "category": "Technology",
            "tags": ["AI", "healthcare", "machine learning", "medical technology"]
        },
        {
            "title": "Remote Work Productivity Study",
            "content": """The global shift to remote work has fundamentally changed how we approach productivity and work-life balance. Recent studies indicate that remote workers are 13% more productive than their office-based counterparts. Key factors contributing to this increase include reduced commuting time, fewer office distractions, and flexible scheduling. However, remote work also presents challenges such as social isolation, communication barriers, and difficulty maintaining team cohesion. Successful remote work strategies include regular video conferences, clear communication protocols, dedicated workspace setup, and maintaining healthy boundaries between work and personal life.""",
            "author": "Lisa Wang",
            "category": "Business",
            "tags": ["remote work", "productivity", "workplace", "efficiency"]
        },
        {
            "title": "The Future of Electric Vehicles",
            "content": """Electric vehicles represent a paradigm shift in transportation technology. With advancing battery technology, EVs now offer longer ranges, faster charging times, and lower operating costs compared to traditional gasoline vehicles. Major automakers are investing billions in electric vehicle development and production. Government incentives and environmental regulations are accelerating adoption rates worldwide. The charging infrastructure is rapidly expanding, addressing one of the primary concerns about EV ownership. By 2030, experts predict that electric vehicles will comprise 30% of all new car sales globally, marking a significant milestone in sustainable transportation.""",
            "author": "Robert Martinez",
            "category": "Transportation",
            "tags": ["electric vehicles", "sustainability", "technology", "automotive"]
        },
        {
            "title": "Mental Health Awareness in the Digital Age",
            "content": """Mental health awareness has gained unprecedented attention in recent years, particularly as digital technology impacts our daily lives. Social media platforms, while connecting people globally, have been linked to increased rates of anxiety and depression, especially among young adults. The constant comparison with others online and fear of missing out contribute to mental health challenges. However, technology also offers solutions through mental health apps, online therapy platforms, and AI-powered support systems. The key is finding balance and using technology mindfully to support rather than hinder mental wellbeing.""",
            "author": "Dr. Emma Thompson",
            "category": "Health",
            "tags": ["mental health", "digital wellness", "social media", "therapy"]
        },
        {
            "title": "Sustainable Agriculture Practices",
            "content": """Sustainable agriculture is essential for feeding the growing global population while protecting environmental resources. Precision farming techniques use GPS, sensors, and data analytics to optimize crop yields while minimizing resource usage. Crop rotation, cover cropping, and integrated pest management reduce the need for chemical fertilizers and pesticides. Vertical farming and hydroponics offer solutions for urban agriculture and water-scarce regions. These practices not only improve environmental sustainability but also enhance soil health, biodiversity, and long-term agricultural productivity. Farmers worldwide are adopting these methods to ensure food security for future generations.""",
            "author": "Professor James Wilson",
            "category": "Agriculture",
            "tags": ["sustainable farming", "agriculture", "food security", "environment"]
        },
        {
            "title": "Blockchain Technology Applications",
            "content": """Blockchain technology extends far beyond cryptocurrency applications. Its decentralized, immutable ledger system offers solutions for supply chain transparency, digital identity verification, and secure data sharing. In healthcare, blockchain can secure patient records while enabling seamless data sharing between providers. Financial services use blockchain for faster, more secure transactions and smart contracts. The technology also shows promise in voting systems, intellectual property protection, and carbon credit trading. Despite challenges like energy consumption and scalability, blockchain continues to evolve with new consensus mechanisms and applications across various industries.""",
            "author": "Alex Kumar",
            "category": "Technology",
            "tags": ["blockchain", "cryptocurrency", "decentralization", "security"]
        },
        {
            "title": "The Psychology of Consumer Behavior",
            "content": """Understanding consumer behavior is crucial for businesses to effectively market their products and services. Psychological factors such as perception, motivation, and social influence significantly impact purchasing decisions. The rise of e-commerce has created new patterns in consumer behavior, with online reviews, social proof, and personalized recommendations playing key roles. Neuromarketing techniques use brain imaging to understand subconscious consumer responses to advertising. Cultural differences, generational preferences, and economic conditions also shape consumer choices. Companies that successfully analyze and adapt to these behavioral patterns gain competitive advantages in the marketplace.""",
            "author": "Dr. Rachel Green",
            "category": "Psychology",
            "tags": ["consumer behavior", "marketing", "psychology", "business"]
        },
        {
            "title": "Renewable Energy Market Trends",
            "content": """The renewable energy sector is experiencing unprecedented growth driven by technological advances and environmental concerns. Solar and wind power costs have decreased dramatically, making them competitive with fossil fuels in many markets. Energy storage solutions, particularly battery technology, are addressing the intermittency challenges of renewable sources. Government policies, carbon pricing, and corporate sustainability commitments are accelerating the transition to clean energy. Investment in renewable energy infrastructure reached record levels, with solar photovoltaic and offshore wind leading the growth. This transition is creating new job opportunities while reducing greenhouse gas emissions globally.""",
            "author": "Dr. Maria Rodriguez",
            "category": "Energy",
            "tags": ["renewable energy", "solar power", "wind energy", "sustainability"]
        },
        {
            "title": "Digital Transformation in Education",
            "content": """Digital transformation has revolutionized educational delivery and student engagement. Online learning platforms, virtual classrooms, and interactive educational tools have made learning more accessible and flexible. Artificial intelligence personalizes learning experiences by adapting to individual student needs and learning styles. Gamification techniques increase student motivation and participation. However, the digital divide remains a challenge, with unequal access to technology affecting educational equity. Teachers require ongoing professional development to effectively integrate technology into their pedagogy. The future of education lies in blended learning models that combine digital innovation with traditional teaching methods.""",
            "author": "Prof. David Lee",
            "category": "Education",
            "tags": ["digital learning", "education technology", "online learning", "pedagogy"]
        },
        {
            "title": "Space Exploration and Commercial Space Industry",
            "content": """Space exploration has entered a new era with private companies joining government agencies in advancing human presence beyond Earth. Commercial space companies are reducing launch costs through reusable rocket technology, making space more accessible for scientific research and commercial applications. Satellite constellations provide global internet coverage and Earth observation capabilities. Mars exploration missions are paving the way for future human colonization. Space tourism is becoming reality with suborbital flights now available to civilians. The space economy is rapidly expanding, encompassing satellite communications, space manufacturing, and asteroid mining prospects.""",
            "author": "Dr. Thomas Anderson",
            "category": "Science",
            "tags": ["space exploration", "commercial space", "satellites", "Mars"]
        },
        {
            "title": "Cybersecurity in the Modern World",
            "content": """Cybersecurity threats have evolved dramatically with increasing digitization of business operations and personal activities. Ransomware attacks, data breaches, and social engineering schemes pose significant risks to organizations and individuals. The Internet of Things expands attack surfaces with connected devices often lacking adequate security measures. Artificial intelligence is being deployed both by attackers to create sophisticated threats and by defenders to detect and respond to incidents. Zero-trust security architectures are becoming standard practice, assuming no network or device is inherently trustworthy. Cybersecurity awareness training and robust incident response plans are essential components of modern security strategies.""",
            "author": "Jennifer Park",
            "category": "Cybersecurity",
            "tags": ["cybersecurity", "data protection", "ransomware", "information security"]
        },
        {
            "title": "Genetic Engineering and CRISPR Technology",
            "content": """CRISPR-Cas9 gene editing technology has revolutionized genetic engineering with its precision and accessibility. This tool allows scientists to modify DNA sequences with unprecedented accuracy, opening possibilities for treating genetic diseases, improving crop yields, and advancing medical research. Clinical trials are underway for CRISPR-based treatments for sickle cell disease, cancer, and inherited blindness. However, ethical considerations surrounding human genetic modification, particularly germline editing, continue to spark debate. Regulatory frameworks are evolving to ensure responsible use of gene editing technologies while fostering innovation that could address some of humanity's greatest health challenges.""",
            "author": "Dr. Helen Zhang",
            "category": "Biotechnology",
            "tags": ["CRISPR", "gene editing", "biotechnology", "genetic engineering"]
        },
        {
            "title": "Urban Planning for Smart Cities",
            "content": """Smart cities integrate technology and data analytics to improve urban services and quality of life for residents. Internet of Things sensors monitor air quality, traffic flow, and energy consumption in real-time. Smart transportation systems optimize traffic signals and public transit schedules based on demand patterns. Digital governance platforms enable citizens to access services and participate in decision-making processes. Sustainable urban design incorporates green infrastructure, renewable energy systems, and efficient waste management. The challenge lies in balancing technological innovation with privacy concerns, digital equity, and the need for inclusive urban development that serves all community members.""",
            "author": "Dr. Carlos Mendez",
            "category": "Urban Planning",
            "tags": ["smart cities", "urban planning", "IoT", "sustainable development"]
        },
        {
            "title": "The Economics of Cryptocurrency",
            "content": """Cryptocurrency markets have matured significantly since Bitcoin's inception, with institutional adoption and regulatory clarity driving mainstream acceptance. Central bank digital currencies are being explored by governments worldwide as potential alternatives to traditional monetary systems. Decentralized finance protocols enable lending, borrowing, and trading without traditional financial intermediaries. However, cryptocurrency markets remain highly volatile, influenced by regulatory announcements, technological developments, and market sentiment. Environmental concerns about energy-intensive mining operations have led to the development of more sustainable consensus mechanisms. The long-term economic impact of cryptocurrencies continues to evolve as adoption grows and regulatory frameworks develop.""",
            "author": "Dr. Patricia Williams",
            "category": "Economics",
            "tags": ["cryptocurrency", "Bitcoin", "DeFi", "digital currency"]
        }
    ]
    
    for doc_data in sample_documents:
        try:
            # Generate document ID
            document_id = hashlib.sha256(
                f"{doc_data['title']}{doc_data['content']}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            # Create metadata
            metadata = DocumentMetadata(
                title=doc_data["title"],
                author=doc_data.get("author"),
                tags=doc_data.get("tags", []),
                category=doc_data.get("category"),
                created_date=datetime.now(),
                modified_date=datetime.now()
            )
            
            # Add document to database
            success = db.add_document(document_id, doc_data["title"], doc_data["content"], metadata)
            
            if success:
                logger.info(f"Added document: {doc_data['title']}")
            else:
                logger.error(f"Failed to add document: {doc_data['title']}")
        except Exception as e:
            logger.error(f"Error adding document {doc_data['title']}: {str(e)}")
    
    logger.info("Sample documents added successfully!")

if __name__ == "__main__":
    # Initialize components
    initialize_components()
    
    # Add sample documents
    asyncio.run(add_sample_documents())
    
    # Run the FastMCP server
    app.run()