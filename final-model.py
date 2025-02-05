# -- coding: utf-8 --
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from transformers import MarianMTModel, MarianTokenizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import spacy
import seaborn as sns
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(page_title="Trans-Sustain", layout="wide")

# Load models with caching
@st.cache_data
def load_translation_model():
    return MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en"), \
           MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

# File upload section
st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload File", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read and preprocess data based on file type
    try:
        if uploaded_file.name.endswith("xlsx"):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the dataset to help users identify the data
        st.subheader("Data Preview")
        st.dataframe(data.head())

        # Generalized Text Columns Detection
        text_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(text_columns) == 0:
            st.error("No text columns detected in the uploaded file.")
        else:
            # Ask users to select the column to perform analysis on
            selected_column = st.selectbox("Select a column for analysis", text_columns)

            # Data cleaning (basic)
            def clean_entry(entry):
                return str(entry).strip()

            data[selected_column] = data[selected_column].apply(clean_entry)

            # Translation section (optional)
            st.header("Text Translation")
            if st.checkbox("Show Translation Progress"):
                tokenizer, model = load_translation_model()
                progress_bar = st.progress(0)  # Initialize progress bar
                
                def translate_text(x):
                    progress_bar.progress(100 * data[selected_column].index(x) / len(data[selected_column]))  # Update progress bar
                    return tokenizer.decode(
                        model.generate(**tokenizer(x, return_tensors="pt", truncation=True))[0],
                        skip_special_tokens=True
                    ) if pd.notnull(x) else x

                with st.spinner('Translating content...'):
                    data[f'{selected_column}_translated'] = data[selected_column].apply(translate_text)
                st.success("Translation completed!")

            # Main display
            st.subheader("Processed Data Preview")
            st.dataframe(data.head())

            # Analysis sections
            tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Analysis", "Topic Modeling", "Word Cloud", "Clustering"])

            with tab1:
                st.header("Sentiment Analysis")
                data['sentiment_score'] = data[selected_column].apply(
                    lambda x: TextBlob(str(x)).sentiment.polarity
                )
                data['sentiment_category'] = data['sentiment_score'].apply(
                    lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral'
                )
                st.write(data[[selected_column, 'sentiment_score', 'sentiment_category']])

            with tab2:
                st.header("Topic Modeling")
                vectorizer = CountVectorizer(stop_words='english')
                dtm = vectorizer.fit_transform(data[selected_column])
                lda = LatentDirichletAllocation(n_components=3, random_state=42)
                lda.fit(dtm)

                for idx, topic in enumerate(lda.components_):
                    st.subheader(f"Topic {idx+1}")
                    st.write(", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]))

            with tab3:
                st.header("Word Cloud")
                all_text = " ".join(data[selected_column].dropna())
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            with tab4:
                st.header("Text Clustering")
                # Add a slider to choose the number of clusters
                num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

                # Check if number of clusters is greater than the number of samples
                n_samples = len(data[selected_column].dropna())
                if num_clusters > n_samples:
                    st.error(f"Cannot have more clusters ({num_clusters}) than samples ({n_samples}).")
                else:
                    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = tfidf_vectorizer.fit_transform(data[selected_column])
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    data['cluster'] = kmeans.fit_predict(tfidf_matrix)
                    st.write(data[[selected_column, 'cluster']])

                    # Reduce dimensionality of the data to 2D using PCA for visualization
                    pca = PCA(n_components=2)
                    pca_components = pca.fit_transform(tfidf_matrix.toarray())
                    data['pca1'] = pca_components[:, 0]
                    data['pca2'] = pca_components[:, 1]

                    # Plot the clusters
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=data, palette='viridis', s=100)
                    plt.title(f'Clusters Visualization (n_clusters={num_clusters})')
                    st.pyplot(plt)

            # Download section
            st.sidebar.header("Download Results")
            if st.sidebar.button("Prepare Download"):
                output = data.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(
                    label="Download Processed Data",
                    data=output,
                    file_name='processed_data.csv',
                    mime='text/csv'
                )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("Please upload a file to begin analysis")

# Footer with credits
st.markdown("<br><br><center><i>Made with ❤️ by Samuel, Christianah, Mehdi, and John</i></center>", unsafe_allow_html=True)
