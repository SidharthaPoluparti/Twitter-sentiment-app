# Step 1: Import Libraries
import streamlit as st
import requests
import json

# Step 2: Configure Streamlit page
st.set_page_config(page_title="Twitter Sentiment Analyzer (3-Class)", layout="wide")

# Step 3: App Title and Description
st.title("🎯 Twitter Sentiment Analysis App (3-Class)")
st.markdown("---")
st.markdown("Analyze the sentiment of tweets using Machine Learning (SVM - 3 Classes)")

# Step 4: API endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Step 5: Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Step 6: Input text area
    user_text = st.text_area(
        "Enter tweet text:",
        placeholder="Paste a tweet or any text here...",
        height=150
    )

with col2:
    st.info("ℹ️ **How it works:**\n- Enter tweet text\n- Click 'Analyze Sentiment'\n- Get instant prediction with 3 classes:\n  - Positive\n  - Negative\n  - Neutral")

# Step 7: Analyze button
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if user_text.strip() == "":
        st.error("❌ Please enter some text to analyze")
    else:
        try:
            # Step 8: Send request to API
            response = requests.post(
                API_URL,
                json={"text": user_text},
                timeout=10
            )
            
            if response.status_code == 200:
                # Step 9: Parse response
                result = response.json()
                sentiment = result['sentiment']
                confidence = result['confidence']
                
                # Step 10: Display results with styling
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Create columns for results
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    if sentiment == "Positive":
                        st.success(f"✅ Sentiment: {sentiment}")
                    elif sentiment == "Negative":
                        st.error(f"❌ Sentiment: {sentiment}")
                    else:  # Neutral
                        st.warning(f"⚪ Sentiment: {sentiment}")
                
                with res_col2:
                    st.info(f"🎯 Confidence: {confidence * 100:.1f}%")
                
                with res_col3:
                    confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.warning(f"📈 Confidence Level: {confidence_level}")
                
                # Step 11: Display analyzed text
                st.markdown("---")
                st.subheader("📝 Your Input")
                st.write(user_text)
                
                # Step 12: Display confidence bar
                st.markdown("---")
                st.subheader("📊 Confidence Score")
                st.progress(confidence)
                st.caption(f"{confidence:.2%} confident in this prediction")
                
            else:
                st.error(f"❌ API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure FastAPI is running on http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Step 13: Add example tweets section
st.markdown("---")
st.subheader("📚 Example Tweets to Try:")

examples = [
    "This product is amazing, absolutely love it!",
    "Worst experience ever, totally disappointed",
    "The service was okay, nothing special",
    "Just finished the new movie and it was incredible!",
    "meh, it was alright i guess"
]

for i, example in enumerate(examples, 1):
    if st.button(f"Try Example {i}: {example[:50]}...", key=f"example_{i}"):
        st.session_state['example_text'] = example
