import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Page Configuration
st.set_page_config(
    page_title="FeaturePulse | AI Feedback Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Glassmorphism & Dashboard Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Clean Hero Container */
    .hero-box {
        background: linear-gradient(135deg, #1E1E2F 0%, #12121A 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-desc {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Cards & Containers */
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-billing { background: #FEE2E2; color: #DC2626; }
    .badge-auth { background: #FEF3C7; color: #D97706; }
    .badge-ui { background: #E0E7FF; color: #4338CA; }
    
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Database & Model Caching
@st.cache_resource
def init_vector_db():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="customer_reviews",
        metadata={"hnsw:space": "cosine"}
    )
    df = pd.read_csv("reviews.csv")
    
    if collection.count() == 0:
        docs = df["review_text"].tolist()
        ids = [str(i) for i in df["review_id"].tolist()]
        metas = [
            {
                "rating": int(row["rating"]),
                "category": str(row["category"]),
                "lost_value": int(row["lost_order_value"]),
                "version": str(row["app_version"])
            }
            for _, row in df.iterrows()
        ]
        embeddings = embed_model.encode(docs).tolist()
        collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
        
    return embed_model, collection, df

embed_model, collection, df = init_vector_db()

# 4. Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Engine Controls")
    top_k = st.slider("Feedback Samples (Top K)", min_value=1, max_value=8, value=4)
    
    categories = ["All Categories"] + sorted(df["category"].unique().tolist())
    selected_cat = st.selectbox("Filter by Category", categories)
    
    st.markdown("---")
    st.markdown("### 🤖 Strategy Engine")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter Google AI Studio key for live AI strategies")
    
    st.markdown("---")
    with st.container(border=True):
        st.caption("⚡ **System Architecture**")
        st.markdown("**Vector DB:** ChromaDB (Local)")
        st.markdown("**Embeddings:** 384-dim (MiniLM)")
        st.markdown(f"**Database Size:** {len(df)} customer logs")

# 5. Header Section
st.markdown("""
<div class="hero-box">
    <div class="hero-title">⚡ FeaturePulse: Product Intelligence Dashboard</div>
    <p class="hero-desc">Discover root causes, map customer pain points semantically, and fix revenue-draining bugs with zero-cost solutions.</p>
</div>
""", unsafe_allow_html=True)

# 6. Offline Simple Strategy Generator
def get_low_budget_strategy(category, sample_texts, total_lost_rev):
    cat = category.lower()
    joined = " ".join(sample_texts).lower()
    
    if "billing" in cat or "pay" in joined or "card" in joined:
        return {
            "root_cause": "The payment gateway took too long to respond. The customer clicked 'Pay' multiple times or the app gave up waiting before the bank finished processing.",
            "tech_fix": "1. Disable the 'Pay' button immediately once clicked so users can't tap it twice.\n2. Add a simple order token so if duplicate requests reach the server, only one goes through.",
            "low_budget_fix": "Use free Python background tasks (or Celery with open-source Redis) to check bank callbacks automatically without paying for third-party monitoring.",
            "profit_impact": f"Instantly saves ₹{total_lost_rev:,} in lost orders and prevents users from leaving angry 1-star reviews."
        }
    elif "auth" in cat or "otp" in joined or "login" in joined:
        return {
            "root_cause": "The SMS provider is delaying OTP text messages, or the finger-print/FaceID login crashed on newer phones.",
            "tech_fix": "1. If the SMS OTP doesn't arrive in 30 seconds, show an option to get the OTP via WhatsApp or Email.\n2. If fingerprint login fails, gracefully show standard Password/PIN entry.",
            "low_budget_fix": "Use existing free SMTP email relays or WhatsApp Business free tier instead of expensive enterprise authentication SaaS.",
            "profit_impact": "Prevents 20% of users from abandoning the app on the sign-in screen."
        }
    else:
        return {
            "root_cause": "The app screens are reloading too many times, making the screen lag, freeze, or display unreadable text in dark mode.",
            "tech_fix": "1. Stop re-rendering components that haven't changed.\n2. Set fixed background and text colors so text never blends into dark mode.",
            "low_budget_fix": "Run Google Lighthouse (built into Chrome for free) to see what is slowing down the page instead of paying for costly performance tools.",
            "profit_impact": "Maintains a 4.0+ Play Store rating and stops users from uninstalling within the first week."
        }

def display_strategy_cards(strat):
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 🔍 What Went Wrong (Root Cause)")
            st.write(strat["root_cause"])
        with st.container(border=True):
            st.markdown("#### 🛠️ Direct Fix (Developer Instructions)")
            st.write(strat["tech_fix"])
    with c2:
        with st.container(border=True):
            st.markdown("#### 💡 Free & Open-Source Solution")
            st.write(strat["low_budget_fix"])
        with st.container(border=True):
            st.markdown("#### 📈 Business & Money Saved")
            st.write(strat["profit_impact"])

# 7. Search Bar
query = st.text_input("🔍 Search customer pain points:", placeholder="e.g., money deducted but order cancelled")

if query:
    st.markdown("### 🔎 Search Comparison")
    col1, col2 = st.columns([1, 1], gap="medium")
    
    # Left: Keyword Search
    with col1:
        with st.container(border=True):
            st.markdown("#### 🔤 Traditional Exact-Word Search")
            st.caption("Searches for the exact words typed above.")
            
            filtered_df = df if selected_cat == "All Categories" else df[df["category"] == selected_cat]
            keyword_results = filtered_df[filtered_df["review_text"].str.contains(query, case=False, na=False)]
            
            if not keyword_results.empty:
                for _, row in keyword_results.head(top_k).iterrows():
                    with st.expander(f"Review #{row['review_id']} • {row['rating']}★ • ₹{row['lost_order_value']:,}"):
                        st.write(row['review_text'])
                        st.caption(f"Category: {row['category']} | Version: {row['app_version']}")
            else:
                st.warning(f"0 matches found for '{query}'")
                st.info("Traditional search failed because users used different words to describe the issue.")

    # Right: Semantic Search
    retrieved_texts = []
    retrieved_metas = []
    
    with col2:
        with st.container(border=True):
            st.markdown("#### 🧠 ChromaDB Semantic Vector Search")
            st.caption("Understands conceptual meaning and intent, not just spelling.")
            
            query_vector = embed_model.encode([query]).tolist()
            
            where_filter = None if selected_cat == "All Categories" else {"category": selected_cat}
            
            vector_results = collection.query(
                query_embeddings=query_vector, 
                n_results=top_k,
                where=where_filter
            )
            
            if vector_results["documents"] and len(vector_results["documents"][0]) > 0:
                for i in range(len(vector_results["documents"][0])):
                    doc_text = vector_results["documents"][0][i]
                    doc_id = vector_results["ids"][0][i]
                    doc_meta = vector_results["metadatas"][0][i]
                    distance = vector_results["distances"][0][i]
                    match_pct = round((1 - distance) * 100, 1)
                    
                    retrieved_texts.append(doc_text)
                    retrieved_metas.append(doc_meta)
                    
                    with st.expander(f"Review #{doc_id} • {match_pct}% Match • {doc_meta['rating']}★ • At-Risk: ₹{doc_meta['lost_value']:,}"):
                        st.write(doc_text)
                        st.caption(f"Category: {doc_meta['category']} | Version: {doc_meta['version']}")
            else:
                st.info("No semantic matches found.")

    # 8. Executive KPI Section
    if retrieved_texts:
        st.markdown("---")
        st.markdown("### 📊 Business Impact Scorecard")
        
        total_risk_val = sum([m["lost_value"] for m in retrieved_metas])
        avg_rating = round(sum([m["rating"] for m in retrieved_metas]) / len(retrieved_metas), 1)
        cats = [m["category"] for m in retrieved_metas]
        dominant_cat = max(set(cats), key=cats.count)
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Customer Complaints Clustered", f"{len(retrieved_texts)} Users")
        kpi2.metric("Segment Rating", f"{avg_rating} / 5.0", delta="-Critical" if avg_rating < 2.5 else "Healthy", delta_color="inverse")
        kpi3.metric("Revenue at Risk", f"₹{total_risk_val:,}", delta="Direct Leakage", delta_color="inverse")
        kpi4.metric("Dominant Area", dominant_cat)
        
        st.markdown("### 📋 Executive Action Plan")
        
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key.strip())
                
                # Simple English Prompt
                prompt = f"""
                You are a Product Manager helping a startup.
                Review these user complaints:
                {retrieved_texts}
                
                Category: {dominant_cat}
                Total Money at Risk: ₹{total_risk_val}
                
                Write an action plan in simple, everyday English. Do NOT use heavy corporate jargon or confusing acronyms.
                Use this exact format:
                
                ### 1. What is going wrong?
                (Explain the issue in 2 simple sentences so anyone can understand.)
                
                ### 2. How developers can fix it (Low-cost / Free)
                (Provide 2 practical steps using free open-source code or logic adjustments.)
                
                ### 3. Business & Money Saved
                (Explain how this saves money and keeps users happy in 2 simple sentences.)
                """
                with st.spinner("Generating straightforward action plan..."):
                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt
                    )
                with st.container(border=True):
                    st.markdown(response.text)
            except Exception as e:
                st.error(f"Live API error: {e}. Falling back to internal engine.")
                strat = get_low_budget_strategy(dominant_cat, retrieved_texts, total_risk_val)
                display_strategy_cards(strat)
        else:
            strat = get_low_budget_strategy(dominant_cat, retrieved_texts, total_risk_val)
            display_strategy_cards(strat)
else:
    # Empty State Guidance
    st.info("💡 Try searching for: **'card payment failed'**, **'otp not coming'**, or **'screen freeze'** to see how vector search outperforms keyword search.")