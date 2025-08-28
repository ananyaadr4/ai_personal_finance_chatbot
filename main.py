from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import json
from datetime import datetime, timedelta
import requests
import uuid
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personal Finance Chatbot API", 
    description="AI-powered financial analysis with Ollama integration",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    category: str

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class FinanceStats(BaseModel):
    total_spent: float
    total_transactions: int
    average_transaction: float
    categories_count: int
    category_breakdown: Dict[str, float]
    monthly_breakdown: Dict[str, float]
    top_merchants: Dict[str, int]

class ChartData(BaseModel):
    type: str  # 'pie', 'bar', 'line'
    title: str
    data: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    response: str
    session_id: str
    chart_data: Optional[ChartData] = None

class BudgetAlert(BaseModel):
    category: str
    percentage: float
    amount: float
    alert_type: str  # 'warning' or 'error'

# In-memory storage (use SQLite or database in production)
user_sessions: Dict[str, List[Transaction]] = {}

# Transaction Classification Service
class TransactionClassifier:
    """Classifies transactions into categories using keyword matching"""
    
    def __init__(self):
        self.category_keywords = {
            'Food': [
                'grocery', 'restaurant', 'cafe', 'coffee', 'food', 'pizza', 'burger', 
                'starbucks', 'mcdonalds', 'subway', 'chipotle', 'dominos', 'kfc',
                'taco bell', 'dunkin', 'bakery', 'deli', 'market', 'supermarket'
            ],
            'Transport': [
                'gas', 'uber', 'lyft', 'taxi', 'bus', 'train', 'metro', 'parking', 
                'toll', 'shell', 'exxon', 'bp', 'chevron', 'mobil', 'flight', 
                'airline', 'car rental', 'zipcar'
            ],
            'Entertainment': [
                'netflix', 'spotify', 'movie', 'theater', 'cinema', 'concert', 
                'game', 'steam', 'youtube', 'hulu', 'disney', 'amazon prime',
                'apple music', 'tickets', 'amusement'
            ],
            'Healthcare': [
                'pharmacy', 'doctor', 'hospital', 'medical', 'health', 'dentist', 
                'cvs', 'walgreens', 'rite aid', 'clinic', 'urgent care',
                'prescription', 'medicine'
            ],
            'Shopping': [
                'amazon', 'target', 'walmart', 'costco', 'shop', 'store', 'mall', 
                'clothing', 'shoes', 'best buy', 'home depot', 'lowes',
                'macy', 'nordstrom', 'online'
            ],
            'Utilities': [
                'electric', 'water', 'internet', 'phone', 'cable', 'wifi', 
                'verizon', 'att', 'comcast', 'spectrum', 'utility', 'bill',
                'power', 'gas company'
            ],
            'Housing': [
                'rent', 'mortgage', 'apartment', 'lease', 'property', 'insurance',
                'maintenance', 'repairs', 'landlord'
            ],
            'Education': [
                'school', 'university', 'course', 'book', 'tuition', 'education',
                'learning', 'training', 'certification'
            ]
        }
    
    def classify(self, description: str) -> str:
        """Classify transaction based on description keywords"""
        desc_lower = description.lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in desc_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # Return category with highest score
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'Other'

# Chart Data Generator
class ChartDataGenerator:
    """Generate chart data based on transactions and query type"""
    
    @staticmethod
    def generate_chart_for_query(query: str, transactions: List[Transaction]) -> Optional[ChartData]:
        """Generate appropriate chart based on the user query"""
        
        if not transactions:
            return None
        
        df = pd.DataFrame([t.dict() for t in transactions])
        df['amount_abs'] = df['amount'].abs()
        df['date'] = pd.to_datetime(df['date'])
        
        query_lower = query.lower()
        
        # Category-related queries -> Pie chart
        if any(word in query_lower for word in ['category', 'categories', 'breakdown', 'distribution']):
            return ChartDataGenerator._create_category_pie_chart(df)
        
        # Monthly/trend queries -> Line chart
        elif any(word in query_lower for word in ['month', 'trend', 'time', 'over time', 'monthly']):
            return ChartDataGenerator._create_monthly_line_chart(df)
        
        # Top/biggest queries -> Bar chart
        elif any(word in query_lower for word in ['top', 'biggest', 'largest', 'most', 'highest']):
            if 'merchant' in query_lower or 'store' in query_lower:
                return ChartDataGenerator._create_top_merchants_bar_chart(df)
            else:
                return ChartDataGenerator._create_category_bar_chart(df)
        
        # Default to category pie chart for spending queries
        elif any(word in query_lower for word in ['spend', 'spending', 'spent', 'money', 'budget']):
            return ChartDataGenerator._create_category_pie_chart(df)
        
        return None
    
    @staticmethod
    def _create_category_pie_chart(df: pd.DataFrame) -> ChartData:
        """Create pie chart for category breakdown"""
        category_data = df.groupby('category')['amount_abs'].sum().sort_values(ascending=False)
        
        chart_data = []
        for category, amount in category_data.items():
            chart_data.append({
                'name': category,
                'value': round(amount, 2)
            })
        
        return ChartData(
            type='pie',
            title='Spending by Category',
            data=chart_data
        )
    
    @staticmethod
    def _create_category_bar_chart(df: pd.DataFrame) -> ChartData:
        """Create bar chart for category comparison"""
        category_data = df.groupby('category')['amount_abs'].sum().sort_values(ascending=False).head(8)
        
        chart_data = []
        for category, amount in category_data.items():
            chart_data.append({
                'category': category,
                'amount': round(amount, 2)
            })
        
        return ChartData(
            type='bar',
            title='Top Categories by Spending',
            data=chart_data
        )
    
    @staticmethod
    def _create_monthly_line_chart(df: pd.DataFrame) -> ChartData:
        """Create line chart for monthly trends"""
        df['month'] = df['date'].dt.strftime('%Y-%m')
        monthly_data = df.groupby('month')['amount_abs'].sum().sort_index()
        
        chart_data = []
        for month, amount in monthly_data.items():
            chart_data.append({
                'month': month,
                'amount': round(amount, 2)
            })
        
        return ChartData(
            type='line',
            title='Monthly Spending Trend',
            data=chart_data
        )
    
    @staticmethod
    def _create_top_merchants_bar_chart(df: pd.DataFrame) -> ChartData:
        """Create bar chart for top merchants"""
        merchant_data = df.groupby('description')['amount_abs'].sum().sort_values(ascending=False).head(10)
        
        chart_data = []
        for merchant, amount in merchant_data.items():
            chart_data.append({
                'merchant': merchant[:30] + '...' if len(merchant) > 30 else merchant,
                'amount': round(amount, 2)
            })
        
        return ChartData(
            type='bar',
            title='Top Merchants by Spending',
            data=chart_data
        )

# Ollama AI Integration
class OllamaFinanceAI:
    """AI assistant powered by Ollama for financial analysis"""
    
    def __init__(self, model_name="llama3.2", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = f"{base_url}/api/generate"
        self.chart_generator = ChartDataGenerator()
        
    async def analyze_query(self, query: str, transactions: List[Transaction]) -> tuple[str, Optional[ChartData]]:
        """Analyze user query and provide financial insights with chart data"""
        
        if not transactions:
            return ("I don't have any transaction data to analyze. Please upload your CSV file first to get started with financial insights!", None)
        
        # Generate chart data if applicable
        chart_data = self.chart_generator.generate_chart_for_query(query, transactions)
        
        # Prepare financial context
        context = self._prepare_financial_context(transactions)
        
        # Create specialized prompt for financial analysis
        prompt = self._create_financial_prompt(query, context, chart_data is not None)
        
        try:
            response = await self._call_ollama(prompt)
            formatted_response = self._format_response(response)
            return (formatted_response, chart_data)
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            # Fallback to rule-based analysis
            fallback_response = self._fallback_analysis(query, transactions)
            return (fallback_response, chart_data)
    
    def _prepare_financial_context(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Prepare comprehensive financial context for the AI"""
        
        df = pd.DataFrame([t.dict() for t in transactions])
        df['date'] = pd.to_datetime(df['date'])
        df['amount_abs'] = df['amount'].abs()
        
        # Calculate key metrics
        total_spent = df['amount_abs'].sum()
        transaction_count = len(df)
        avg_transaction = total_spent / transaction_count if transaction_count > 0 else 0
        
        # Category analysis
        category_breakdown = df.groupby('category')['amount_abs'].sum().to_dict()
        category_percentages = {
            cat: (amount / total_spent * 100) if total_spent > 0 else 0 
            for cat, amount in category_breakdown.items()
        }
        
        # Monthly breakdown
        df['month'] = df['date'].dt.strftime('%Y-%m')
        monthly_breakdown = df.groupby('month')['amount_abs'].sum().to_dict()
        
        # Top merchants
        top_merchants = df['description'].value_counts().head(5).to_dict()
        
        # Recent transactions (last 7 days)
        recent_date = df['date'].max() - timedelta(days=7)
        recent_transactions = df[df['date'] > recent_date]
        
        return {
            'total_spent': total_spent,
            'transaction_count': transaction_count,
            'avg_transaction': avg_transaction,
            'category_breakdown': category_breakdown,
            'category_percentages': category_percentages,
            'monthly_breakdown': monthly_breakdown,
            'top_merchants': top_merchants,
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'recent_transaction_count': len(recent_transactions),
            'biggest_expense': df.loc[df['amount_abs'].idxmax()].to_dict() if len(df) > 0 else None
        }
    
    def _create_financial_prompt(self, query: str, context: Dict[str, Any], has_chart: bool) -> str:
        """Create a specialized prompt for financial analysis"""
        
        chart_note = "\nNote: A visual chart is being provided alongside this response." if has_chart else ""
        
        prompt = f"""You are a helpful personal finance assistant. Analyze the user's financial data and provide specific, actionable insights.{chart_note}

FINANCIAL DATA SUMMARY:
- Total Spending: ${context['total_spent']:.2f}
- Number of Transactions: {context['transaction_count']}
- Average Transaction: ${context['avg_transaction']:.2f}
- Date Range: {context['date_range']}

SPENDING BY CATEGORY:
"""
        
        # Add category breakdown
        for category, amount in sorted(context['category_breakdown'].items(), key=lambda x: x[1], reverse=True):
            percentage = context['category_percentages'][category]
            prompt += f"- {category}: ${amount:.2f} ({percentage:.1f}%)\n"
        
        prompt += f"\nTOP MERCHANTS:\n"
        for merchant, count in list(context['top_merchants'].items())[:3]:
            prompt += f"- {merchant}: {count} transactions\n"
        
        if context['biggest_expense']:
            biggest = context['biggest_expense']
            prompt += f"\nBIGGEST EXPENSE: {biggest['description']} - ${biggest['amount_abs']:.2f} ({biggest['category']})\n"
        
        prompt += f"""
USER QUESTION: "{query}"

INSTRUCTIONS:
- Provide specific numbers and calculations
- Be conversational and helpful
- Give actionable insights when possible
- If asking about trends, compare different time periods
- For budget questions, highlight any categories over 25% of spending
- Format monetary amounts with dollar signs and 2 decimal places
- Keep responses concise but informative (2-4 sentences max)
{"- Reference the chart when discussing visual data" if has_chart else ""}

RESPONSE:"""
        
        return prompt
    
    async def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama"""
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 2048,
                "stop": ["USER QUESTION:", "FINANCIAL DATA:", "INSTRUCTIONS:"]
            }
        }
        
        response = requests.post(
            self.base_url, 
            json=data, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response generated')
        else:
            raise Exception(f"Ollama API returned status {response.status_code}: {response.text}")
    
    def _format_response(self, response: str) -> str:
        """Clean and format the AI response"""
        
        # Remove any prompt artifacts
        response = response.strip()
        
        # Remove common AI artifacts
        artifacts_to_remove = [
            "RESPONSE:", "Response:", "ANSWER:", "Answer:",
            "Based on the financial data provided:",
            "Looking at your financial data:"
        ]
        
        for artifact in artifacts_to_remove:
            response = response.replace(artifact, "").strip()
        
        # Ensure proper formatting
        if not response:
            return "I was unable to generate a proper analysis. Please try rephrasing your question."
        
        return response
    
    def _fallback_analysis(self, query: str, transactions: List[Transaction]) -> str:
        """Fallback analysis when Ollama is unavailable"""
        
        df = pd.DataFrame([t.dict() for t in transactions])
        df['amount_abs'] = df['amount'].abs()
        df['date'] = pd.to_datetime(df['date'])
        
        query_lower = query.lower()
        
        # Simple pattern matching for common queries
        if 'food' in query_lower and ('spend' in query_lower or 'spent' in query_lower):
            food_spending = df[df['category'] == 'Food']['amount_abs'].sum()
            return f"You spent ${food_spending:.2f} on food based on your transaction data."
        
        elif 'category' in query_lower:
            category_spending = df.groupby('category')['amount_abs'].sum().sort_values(ascending=False)
            response = "Here's your spending breakdown by category:\n\n"
            for category, amount in category_spending.head(5).items():
                percentage = (amount / category_spending.sum()) * 100
                response += f"• {category}: ${amount:.2f} ({percentage:.1f}%)\n"
            return response
        
        elif 'biggest' in query_lower or 'largest' in query_lower:
            biggest = df.nlargest(3, 'amount_abs')
            response = "Your biggest expenses:\n\n"
            for i, (_, row) in enumerate(biggest.iterrows(), 1):
                response += f"{i}. {row['description']}: ${row['amount_abs']:.2f} ({row['category']})\n"
            return response
        
        else:
            total = df['amount_abs'].sum()
            count = len(df)
            categories = df['category'].nunique()
            return f"""You spent ${total:.2f} across {count} transactions in {categories} categories. Your average transaction was ${total/count:.2f}. Ask me something specific like "How much did I spend on food?" or "Show me my biggest expenses"."""

# Initialize services
classifier = TransactionClassifier()
finance_ai = OllamaFinanceAI()

# Utility Functions
def calculate_stats(transactions: List[Transaction]) -> FinanceStats:
    """Calculate comprehensive financial statistics"""
    
    if not transactions:
        return FinanceStats(
            total_spent=0, total_transactions=0, average_transaction=0,
            categories_count=0, category_breakdown={}, monthly_breakdown={},
            top_merchants={}
        )
    
    df = pd.DataFrame([t.dict() for t in transactions])
    df['amount_abs'] = df['amount'].abs()
    df['date'] = pd.to_datetime(df['date'])
    
    # Basic stats
    total_spent = df['amount_abs'].sum()
    total_transactions = len(df)
    average_transaction = total_spent / total_transactions if total_transactions > 0 else 0
    categories_count = df['category'].nunique()
    
    # Category breakdown
    category_breakdown = df.groupby('category')['amount_abs'].sum().to_dict()
    
    # Monthly breakdown
    df['month'] = df['date'].dt.strftime('%Y-%m')
    monthly_breakdown = df.groupby('month')['amount_abs'].sum().to_dict()
    
    # Top merchants
    top_merchants = df['description'].value_counts().head(10).to_dict()
    
    return FinanceStats(
        total_spent=total_spent,
        total_transactions=total_transactions,
        average_transaction=average_transaction,
        categories_count=categories_count,
        category_breakdown=category_breakdown,
        monthly_breakdown=monthly_breakdown,
        top_merchants=top_merchants
    )

def generate_budget_alerts(stats: FinanceStats) -> List[BudgetAlert]:
    """Generate budget alerts based on spending patterns"""
    
    alerts = []
    total_spent = stats.total_spent
    
    if total_spent == 0:
        return alerts
    
    for category, amount in stats.category_breakdown.items():
        percentage = (amount / total_spent) * 100
        
        if percentage > 35:
            alerts.append(BudgetAlert(
                category=category,
                percentage=percentage,
                amount=amount,
                alert_type='error'
            ))
        elif percentage > 25:
            alerts.append(BudgetAlert(
                category=category,
                percentage=percentage,
                amount=amount,
                alert_type='warning'
            ))
    
    return alerts

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Personal Finance Chatbot API is running!",
        "version": "1.0.0",
        "ai_backend": "Ollama",
        "endpoints": ["/upload-transactions", "/chat", "/stats", "/health"]
    }

@app.get("/health")
async def health_check():
    """Detailed health check including Ollama status"""
    
    # Check Ollama availability
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = "online" if response.status_code == 200 else "offline"
        ollama_models = response.json().get("models", []) if response.status_code == 200 else []
    except:
        ollama_status = "offline"
        ollama_models = []
    
    return {
        "status": "healthy",
        "ollama_status": ollama_status,
        "available_models": [model.get("name", "unknown") for model in ollama_models],
        "active_sessions": len(user_sessions)
    }

@app.post("/upload-transactions")
async def upload_transactions(file: UploadFile = File(...)):
    """Upload and process transaction CSV file"""
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read and validate file content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['date', 'description', 'amount']
        df_columns_lower = [col.lower() for col in df.columns]
        missing_columns = [col for col in required_columns if col not in df_columns_lower]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}. Required: date, description, amount"
            )
        
        # Normalize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['date']:
                column_mapping[col] = 'date'
            elif col_lower in ['description', 'desc', 'merchant', 'vendor']:
                column_mapping[col] = 'description'
            elif col_lower in ['amount', 'amt', 'value']:
                column_mapping[col] = 'amount'
            elif col_lower in ['category', 'cat', 'type']:
                column_mapping[col] = 'category'
        
        df = df.rename(columns=column_mapping)
        
        # Process transactions
        transactions = []
        processed_count = 0
        
        for _, row in df.iterrows():
            try:
                # Parse date
                date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                
                # Parse amount
                amount = float(row['amount'])
                
                # Get or classify category
                category = row.get('category', None)
                if pd.isna(category) or not category:
                    category = classifier.classify(str(row['description']))
                
                transaction = Transaction(
                    date=date_str,
                    description=str(row['description']).strip(),
                    amount=amount,
                    category=str(category).strip()
                )
                transactions.append(transaction)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Skipped invalid transaction row: {e}")
                continue
        
        if processed_count == 0:
            raise HTTPException(status_code=400, detail="No valid transactions found in CSV")
        
        # Generate session ID and store transactions
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = transactions
        
        # Calculate statistics and alerts
        stats = calculate_stats(transactions)
        alerts = generate_budget_alerts(stats)
        
        logger.info(f"Successfully processed {processed_count} transactions for session {session_id}")
        
        return {
            "message": f"Successfully processed {processed_count} transactions",
            "session_id": session_id,
            "transaction_count": processed_count,
            "stats": stats.dict(),
            "alerts": [alert.dict() for alert in alerts],
            "categories_detected": list(set(t.category for t in transactions))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/chat")
async def chat_with_ai(message: ChatMessage):
    """Chat with AI about finances using Ollama with chart integration"""
    
    session_id = message.session_id
    
    if not session_id or session_id not in user_sessions:
        return ChatResponse(
            response="I don't have access to your transaction data. Please upload a CSV file first to get started with financial analysis!",
            session_id=session_id or "no-session"
        )
    
    transactions = user_sessions[session_id]
    logger.info(f"Processing chat query for session {session_id}: {message.message}")
    
    try:
        response_text, chart_data = await finance_ai.analyze_query(message.message, transactions)
        return ChatResponse(
            response=response_text, 
            session_id=session_id,
            chart_data=chart_data
        )
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        return ChatResponse(
            response="I'm experiencing some technical difficulties. Please try again in a moment.",
            session_id=session_id
        )

@app.get("/stats/{session_id}")
async def get_stats(session_id: str):
    """Get financial statistics for a session"""
    
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    transactions = user_sessions[session_id]
    stats = calculate_stats(transactions)
    alerts = generate_budget_alerts(stats)
    
    return {
        "stats": stats.dict(),
        "alerts": [alert.dict() for alert in alerts]
    }

@app.get("/transactions/{session_id}")
async def get_transactions(session_id: str, limit: int = 100):
    """Get transactions for a session with optional limit"""
    
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    transactions = user_sessions[session_id]
    limited_transactions = transactions[:limit] if limit > 0 else transactions
    
    return {
        "transactions": [t.dict() for t in limited_transactions],
        "total_count": len(transactions),
        "returned_count": len(limited_transactions)
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data"""
    
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del user_sessions[session_id]
    return {"message": f"Session {session_id} deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)