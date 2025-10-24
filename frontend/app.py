import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Financial Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral { color: #95a5a6; }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if backend API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy", data
        return False, None
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False, None


def fetch_news(ticker: str):
    """Fetch news data from backend API"""
    try:
        with st.spinner(f"Fetching latest news for {ticker}..."):
            response = requests.get(
                f"{API_BASE_URL}/news/{ticker}",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                st.error(f"Invalid request: {response.json().get('detail', 'Unknown error')}")
            elif response.status_code == 503:
                st.error("Cannot connect to news API. Please check your API key.")
            elif response.status_code == 504:
                st.error("Request timed out. Please try again.")
            else:
                st.error(f"Error fetching news: HTTP {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Please ensure the backend service is running.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        logger.error(f"Error fetching news: {e}")
        return None


def fetch_stats(ticker: str):
    """Fetch sentiment statistics from backend API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/news/{ticker}/stats",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        return None
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return None


def get_sentiment_color(label: str) -> str:
    """Get color for sentiment label"""
    colors = {
        "positive": "#2ecc71",
        "negative": "#e74c3c",
        "neutral": "#95a5a6"
    }
    return colors.get(label.lower(), "#95a5a6")


def get_sentiment_emoji(label: str) -> str:
    """Get emoji for sentiment label"""
    emojis = {
        "positive": "📈",
        "negative": "📉",
        "neutral": "➡️"
    }
    return emojis.get(label.lower(), "➡️")


def create_sentiment_distribution_chart(stats: dict):
    """Create pie chart for sentiment distribution"""
    if not stats or stats.get("total_news", 0) == 0:
        return None
    
    labels = ["Positive", "Negative", "Neutral"]
    values = [
        stats.get("positive_count", 0),
        stats.get("negative_count", 0),
        stats.get("neutral_count", 0)
    ]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_confidence_chart(items: list):
    """Create bar chart for confidence levels"""
    if not items:
        return None
    
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    
    for item in items:
        confidence = item.get("ml_sentiment", {}).get("confidence", "").lower()
        if confidence in confidence_counts:
            confidence_counts[confidence] += 1
    
    fig = go.Figure(data=[go.Bar(
        x=["High", "Medium", "Low"],
        y=[confidence_counts["high"], confidence_counts["medium"], confidence_counts["low"]],
        marker_color=["#2ecc71", "#f39c12", "#e74c3c"],
        text=[confidence_counts["high"], confidence_counts["medium"], confidence_counts["low"]],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Prediction Confidence Levels",
        xaxis_title="Confidence",
        yaxis_title="Count",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_timeline_chart(items: list):
    """Create timeline chart of sentiment over time"""
    if not items:
        return None
    
    # Prepare data
    df_data = []
    for item in items:
        df_data.append({
            "time": datetime.fromisoformat(item["time_published"].replace("Z", "")),
            "sentiment": item["ml_sentiment"]["label"],
            "score": item["ml_sentiment"]["score"],
            "title": item["title"][:50] + "..."
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values("time")
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x="time",
        y="score",
        color="sentiment",
        color_discrete_map={
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#95a5a6"
        },
        hover_data=["title"],
        size_max=15
    )
    
    fig.update_layout(
        title="Sentiment Timeline",
        xaxis_title="Time",
        yaxis_title="Confidence Score",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def display_news_item(item: dict, index: int):
    """Display individual news item"""
    ml_sentiment = item.get("ml_sentiment", {})
    sentiment_label = ml_sentiment.get("label", "neutral")
    sentiment_score = ml_sentiment.get("score", 0)
    confidence = ml_sentiment.get("confidence", "unknown")
    
    # Sentiment badge
    emoji = get_sentiment_emoji(sentiment_label)
    color = get_sentiment_color(sentiment_label)
    
    with st.container():
        col1, col2, col3 = st.columns([6, 2, 2])
        
        with col1:
            st.markdown(f"**{index}. {item['title']}**")
            st.caption(f"🏢 {item['source']} | 🕐 {item['time_published'][:10]}")
            with st.expander("Read summary"):
                st.write(item['summary'])
                st.markdown(f"[Read full article]({item['url']})")
        
        with col2:
            st.markdown(
                f"<div style='text-align: center; padding: 10px; background-color: {color}20; border-radius: 5px;'>"
                f"<span style='font-size: 24px;'>{emoji}</span><br>"
                f"<span style='color: {color}; font-weight: bold;'>{sentiment_label.upper()}</span><br>"
                f"<span style='font-size: 12px;'>Score: {sentiment_score:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col3:
            st.metric("Relevance", f"{item['relevance_score']:.2f}")
            st.caption(f"Confidence: {confidence}")
        
        st.divider()


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Financial Sentiment Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Real-time sentiment analysis of AAPL financial news powered by AI**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Health Check
        st.subheader("System Status")
        health_status, health_data = check_api_health()
        
        if health_status:
            st.success("✅ System Healthy")
            if health_data:
                st.caption(f"Model: {'✅' if health_data.get('model_loaded') else '❌'}")
                st.caption(f"API: {'✅' if health_data.get('api_accessible') else '❌'}")
        else:
            st.error("❌ System Unavailable")
            st.warning("Please check that the backend service is running.")
            return
        
        st.divider()
        
        # Ticker selection (fixed to AAPL for this version)
        st.subheader("Stock Selection")
        ticker = st.text_input("Ticker Symbol", value="AAPL", disabled=True)
        st.caption("Currently analyzing: Apple Inc.")
        
        st.divider()
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Information
        st.subheader("ℹ️ About")
        st.caption("This dashboard fetches real-time financial news and analyzes sentiment using FinBERT, a pre-trained NLP model fine-tuned for financial sentiment analysis.")
        
        st.divider()
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📰 Latest News for {ticker}")
    
    with col2:
        if st.button("↻ Refresh News", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch data
    news_data = fetch_news(ticker)
    
    if not news_data:
        st.warning("No data available. Please check your API configuration and try again.")
        return
    
    items = news_data.get("items", [])
    
    if not items:
        st.info("No recent news found for this ticker.")
        return
    
    # Fetch statistics
    stats = fetch_stats(ticker)
    
    # Display metrics
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total News", news_data.get("total_items", 0))
    
    with col2:
        if stats:
            st.metric(
                "Positive",
                stats.get("positive_count", 0),
                delta=f"{stats.get('sentiment_distribution', {}).get('positive', 0):.1f}%"
            )
    
    with col3:
        if stats:
            st.metric(
                "Negative",
                stats.get("negative_count", 0),
                delta=f"{stats.get('sentiment_distribution', {}).get('negative', 0):.1f}%",
                delta_color="inverse"
            )
    
    with col4:
        if stats:
            st.metric(
                "Neutral",
                stats.get("neutral_count", 0),
                delta=f"{stats.get('sentiment_distribution', {}).get('neutral', 0):.1f}%",
                delta_color="off"
            )
    
    with col5:
        if stats:
            avg_sentiment = stats.get("avg_sentiment_score", 0)
            sentiment_indicator = "📈" if avg_sentiment > 0 else "📉" if avg_sentiment < 0 else "➡️"
            st.metric(
                "Avg Sentiment",
                f"{avg_sentiment:.2f} {sentiment_indicator}"
            )
    
    st.divider()
    
    # Visualizations
    st.subheader("📈 Analytics")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        if stats:
            fig_pie = create_sentiment_distribution_chart(stats)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
        
        fig_timeline = create_timeline_chart(items)
        if fig_timeline:
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    with viz_col2:
        fig_confidence = create_confidence_chart(items)
        if fig_confidence:
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Top sources
        st.subheader("📰 Top News Sources")
        sources = {}
        for item in items:
            source = item.get("source", "Unknown")
            sources[source] = sources.get(source, 0) + 1
        
        source_df = pd.DataFrame(
            list(sources.items()),
            columns=["Source", "Count"]
        ).sort_values("Count", ascending=False).head(5)
        
        st.dataframe(source_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # News feed
    st.subheader("📋 News Feed")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        sentiment_filter = st.multiselect(
            "Filter by Sentiment",
            ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"]
        )
    
    with filter_col2:
        confidence_filter = st.multiselect(
            "Filter by Confidence",
            ["high", "medium", "low"],
            default=["high", "medium", "low"]
        )
    
    # Filter items
    filtered_items = [
        item for item in items
        if item.get("ml_sentiment", {}).get("label", "").lower() in sentiment_filter
        and item.get("ml_sentiment", {}).get("confidence", "").lower() in confidence_filter
    ]
    
    st.caption(f"Showing {len(filtered_items)} of {len(items)} news items")
    
    # Display news items
    for idx, item in enumerate(filtered_items, 1):
        display_news_item(item, idx)
    
    # Footer
    st.divider()
    st.caption("Powered by FastAPI, Streamlit, and Hugging Face FinBERT | Data from Alpha Vantage")


if __name__ == "__main__":
    main()