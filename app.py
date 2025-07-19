import streamlit as st
import pandas as pd
import tempfile
import os
from langchain_helper import create_vector_db, get_qa_chain

# Page config
st.set_page_config(
    page_title="AskVault 🧠",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>       
    .block-container{
        padding-top: 0;
    }     
            
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 0rem;
    }
    
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0rem;
        padding-top: 0;
        margin: 0 0 0.5rem 0;
    }
    
    .demo-section {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    
    .behind-scenes {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #dee2e6;
        color: #6c757d;
    }
    
    .social-links {
        margin-top: 0.5rem;
    }
    
    .social-links a {
        margin: 0 10px;
        text-decoration: none;
        font-size: 1.2rem;
    }
    
    .progress-indicator {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: #e9ecef;
        border-radius: 8px;
    }
    
    .progress-step {
        flex: 1;
        text-align: center;
        padding: 0.3rem;
        border-radius: 5px;
        margin: 0 0.2rem;
    }
    
    .progress-step.active {
        background: #007bff;
        color: white;
    }
    
    .progress-step.completed {
        background: #28a745;
        color: white;
    }
    
    /* Reduce spacing between elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact paragraphs */
    p {
        margin-bottom: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_question_input' not in st.session_state:
    st.session_state.current_question_input = ""
if 'vector_db_created' not in st.session_state:
    st.session_state.vector_db_created = False
if 'csv_uploaded' not in st.session_state:
    st.session_state.csv_uploaded = False

# Check if vector database exists
import os
if os.path.exists("faiss_index"):
    st.session_state.vector_db_created = True

# 1. Main AskVault Heading
st.markdown("""
<div class="main-header">
    <h1>AskVault 🧠</h1>
    <p>Find answers from your data, not from thin air.</p>
</div>
""", unsafe_allow_html=True)

# 2. What is AskVault section
st.markdown("""
<div class="section-header">
    <h2>🤔 What is AskVault?</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Some organizations like EdTech platforms, e-commerce sites, and customer support teams, maintain large FAQ knowledge bases. But simply adding ChatGPT or any AI chatbot and feeding it a big CSV often leads to hallucinated answers.

**AskVault** solves this by using a **vector database with smart retrieval**, ensuring the AI focuses only on the most relevant information to answer accurately.
""")

# 3. Try the Demo First section
st.markdown("""
<div class="section-header">
    <h2>🎯 Try the Demo First!</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="demo-section">
    <h3>Demo Dataset: Codebasics EdTech FAQ</h3>
    <p>You don’t need your own CSV to try AskVault. This demo uses a knowledge base inspired by this excellent <a href="https://www.youtube.com/watch?v=AjQPRomyd-k">Codebasics YouTube tutorial</a>, based on an EdTech organization where people ask questions about courses, pricing, internships, and more..</p>
</div>
""", unsafe_allow_html=True)

# Show demo is ready
if st.session_state.vector_db_created:
    status_text = "✅ Knowledge base is ready!"
    if st.session_state.csv_uploaded:
        status_text += " (Using your custom dataset)"
    else:
        status_text += " (Using Codebasics FAQ dataset)"
    st.success(status_text)
else:
    st.info("👆 Please upload a CSV file to start asking questions, or manually create the demo database by placing 'codebasics_faqs.csv' in the project folder and running the setup.")

# Sample questions
if st.session_state.vector_db_created:
    st.markdown("**Try these sample questions:**")
    predefined_questions = [
        "Do you offer EMI options?",
        "How long will the course take?",
        "Is this course for beginners?",
        "Do you provide internships?",
        "What is the refund policy?",
        "Do you provide job assistance?"
    ]
    
    cols = st.columns(6)
    for i, pre_q in enumerate(predefined_questions):
        with cols[i]:
            if st.button(pre_q, key=f"demo_btn_{i}"):
                st.session_state.current_question_input = pre_q
                st.rerun()

# 4. Ask Your Question section
st.markdown("""
<div class="section-header">
    <h2>💬 Ask Your Question</h2>
</div>
""", unsafe_allow_html=True)

# Question input
question = st.text_input(
    "Enter your question:", 
    value=st.session_state.current_question_input, 
    key="input_field",
    placeholder="Type your question here..."
)

# CSV Download section
st.markdown("""
[📁 Download sample CSV](https://github.com/harsh-c137/Ask-Vault/blob/main/codebasics_faqs.csv) to see the required format.
""")

# Submit and clear buttons
col1, col2 = st.columns([1, 4])
with col1:
    submit_clicked = st.button("🔍 Get Answer", type="primary")
with col2:
    if st.button("🗑️ Clear Question"):
        st.session_state.current_question_input = ""
        st.rerun()

# Store the answer and source documents for later display
answer_content = None
source_documents = None

# Process question
if (submit_clicked or question) and question.strip():
    with st.spinner("Searching knowledge base and generating answer..."):
        try:
            raw_llm_response = get_qa_chain(question)
            answer_content = raw_llm_response['result']
            source_documents = raw_llm_response.get('source_documents', [])
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            st.info("Please try rephrasing your question or check if the knowledge base is properly set up.")

# 5. Answer section (only shown if we have an answer)
if answer_content:
    st.markdown("""
    <div class="section-header">
        <h2>✅ Answer</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: #d1ecf1; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #17a2b8;">
        <h3>Response:</h3>
        <p style="font-size: 1.1rem; line-height: 1.6;">{answer_content}</p>
    </div>
    """, unsafe_allow_html=True)

# 6. Behind the Scenes section (only shown if we have source documents)
if source_documents:
    st.markdown("""
    <div class="section-header">
        <h2>🔍 Behind the Scenes</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="behind-scenes">
        <p>Here are the relevant Q&As our AI retrieved from the knowledge base before generating the final answer. 
        This transparency helps you understand how the response was constructed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show source documents
    with st.expander("📚 View Retrieved Context", expanded=False):
        for i, doc in enumerate(source_documents):
            st.markdown(f"""
            **Source {i+1}:**
            - **Question:** {doc.page_content}
            - **Metadata:** {doc.metadata}
            """)

# 7. Upload Your Own CSV section
st.markdown("""
<div class="section-header">
    <h2>📤 Upload Your Own CSV</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="upload-section">
    <h3>Ready to use your own data?</h3>
    <p>Upload your CSV file with 'question' and 'answer' columns to create your custom knowledge base.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Validate CSV format
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['prompt', 'response']
        
        if all(col in df.columns for col in required_columns):
            st.success(f"✅ CSV format validated! Found {len(df)} Q&A pairs.")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                temp_file_path = tmp_file.name
            
            if st.button("🔄 Create Knowledge Base from Your CSV", type="primary"):
                with st.spinner("Processing your CSV and creating knowledge base..."):
                    try:
                        # Create vector DB from uploaded file
                        create_vector_db(temp_file_path)
                        st.session_state.vector_db_created = True
                        st.session_state.csv_uploaded = True
                        st.success("Your knowledge base is ready!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating knowledge base: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                            
        else:
            st.error(f"❌ CSV must contain columns: {required_columns}. Found: {list(df.columns)}")
            st.info("Please ensure your CSV has 'prompt' and 'response' columns.")
            
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")

# Reset button
if st.session_state.vector_db_created and st.session_state.csv_uploaded:
    if st.button("🔄 Reset to Demo Database"):
        with st.spinner("Resetting to demo database..."):
            try:
                create_vector_db()  # Reload demo database
                st.session_state.csv_uploaded = False
                st.session_state.current_question_input = ""
                st.success("Reset to demo database!")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting database: {str(e)}")

# 8. How It Works section
st.markdown("""
<div class="section-header">
    <h2>⚙️ How It Works</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **1. Upload your FAQ CSV**
    - Required columns: 'prompt', 'response'
    - UTF-8 encoding
    - No special characters in headers
    """)
with col2:
    st.markdown("""
    **2. Ask any question**
    - Natural language queries
    - The AI searches your knowledge base
    - Finds most relevant Q&As
    """)
with col3:
    st.markdown("""
    **3. Get accurate answers**
    - Source-backed responses
    - No hallucinations
    - Complete transparency
    """)

# 9. Impressed with AskVault section
st.markdown("""
<div class="footer">
    <h3>Like this movie recommender? 🚀</h3>
    <p>Let's collaborate on your next AI project! I specialize in building intelligent applications that solve real business problems.</p>
    <div class="social-links">
        <a href="https://www.github.com/harsh-c137" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" alt="GitHub Icon" width="20" height="20"> GitHub
        </a>
        <a href="https://www.linkedin.com/in/harsh-deshpande-v1/" target="_blank">
            <img src="https://www.svgrepo.com/show/157006/linkedin.svg" alt="LinkedIn Icon" width="20" height="20"> LinkedIn
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
