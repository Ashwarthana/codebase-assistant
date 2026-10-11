import pytest
from typing import List, Dict, Any

class MockGenerator:
    def __init__(self, response_text: str):
        self.response_text = response_text
        
    def generate(self, prompt: str) -> str:
        return self.response_text


@pytest.fixture
def mock_context_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "id": "chunk_1",
            "file": "app/auth/views.py",
            "start_line": 10,
            "end_line": 20,
            "code": "def login(request):\n    \"\"\"\n    Handle user login.\n    \"\"\"\n    username = request.form.get('username')\n    password = request.form.get('password')\n    user = authenticate(username, password)\n    if user:\n        login_user(user)\n        return redirect(url_for('home'))\n    return render_template('login.html', error='Invalid credentials')",
            "type": "function",
            "name": "login"
        },
        {
            "id": "chunk_2",
            "file": "app/models.py",
            "start_line": 5,
            "end_line": 50,
            "code": "class User(db.Model):\n    \"\"\"\n    User model representing a registered user.\n    \"\"\"\n    __tablename__ = 'users'\n    id = db.Column(db.Integer, primary_key=True)\n    username = db.Column(db.String(80), unique=True, nullable=False)\n    email = db.Column(db.String(120), unique=True, nullable=False)\n    password_hash = db.Column(db.String(128))\n\n    def set_password(self, password):\n        self.password_hash = generate_password_hash(password)\n\n    def check_password(self, password):\n        return check_password_hash(self.password_hash, password)",
            "type": "class",
            "name": "User"
        },
        {
            "id": "chunk_3",
            "file": "app/utils/email.py",
            "start_line": 1,
            "end_line": 15,
            "code": "import smtplib\nfrom email.message import EmailMessage\n\ndef send_email(to_address: str, subject: str, body: str) -> bool:\n    \"\"\"\n    Send an email to the specified address.\n    \"\"\"\n    msg = EmailMessage()\n    msg.set_content(body)\n    msg['Subject'] = subject\n    msg['From'] = 'noreply@example.com'\n    msg['To'] = to_address\n    try:\n        # Send email logic\n        return True\n    except Exception:\n        return False",
            "type": "function",
            "name": "send_email"
        }
    ]

class MockRetrieverClass:
    def __init__(self, chunks):
        self.chunks = chunks

    def hybrid_retrieve(self, query: str, k: int = 5) -> List[tuple]:
        # Return chunks with dummy scores
        return [(chunk, 0.9) for chunk in self.chunks[:k]]

    def semantic_search(self, query: str, k: int = 5) -> List[tuple]:
        return [(chunk, 0.8) for chunk in self.chunks[:k]]

    def expand_with_call_graph(self, top_chunks: List[tuple], hops: int = 1) -> List[Dict[str, Any]]:
        return self.chunks


@pytest.fixture
def mock_retriever(mock_context_chunks):
    return MockRetrieverClass(mock_context_chunks)
