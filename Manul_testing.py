import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
import time

# Load and cache the pre-trained model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to categorize semantic deviation based on similarity percentage
def categorize_semantic_deviation(similarity_percentage):
    if similarity_percentage >= 90:
        return "Matched"
    elif similarity_percentage >= 70:
        return "Moderate Review"
    elif similarity_percentage >= 50:
        return "Need Review"
    elif similarity_percentage >= 30:
        return "Significant Review"
    else:
        return "Not Matched"

# Function to create a downloadable Excel file
def create_download_link(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Similarity Results')
    output.seek(0)
    return output

# Main function to define app layout
def main():
    st.title("Text Similarity Analysis and Categorization")
    
    # Main layout with two columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Navigation")
        options = [
            "Home", "Upload Data", "Manual Input"
        ]
        choice = st.radio("Go to", options)
    
    if choice == "Home":
        # Home Page Content
        st.markdown("""
        <h2 style='font-size:28px;'>Semantic Similarity</h2>
        <p style='font-size:16px;'>Measures how similar two sentences are in meaning using models (e.g., Sentence Transformers).</p>
        <li><strong>Mean Similarity (%):</strong> Mean similarity expressed as a percentage.</li>
        <li><strong>Semantic Similarity (%):</strong> Semantic similarity score as a percentage.</li>
        <h2 style='font-size:28px;'>Semantic Deviation Categories</h2>
        <ul style='font-size:16px;'>
        <li><strong>Matched:</strong> 90% to 100%</li>
        <li><strong>Moderate Review:</strong> 70% to 89.99%</li>
        <li><strong>Need Review:</strong> 50% to 69.99%</li>
        <li><strong>Significant Review:</strong> 30% to 49.99%</li>
        <li><strong>Not Matched:</strong> 0% to 29.99%</li>
        </ul>
        """, unsafe_allow_html=True)

    elif choice == "Upload Data":
        # Upload Excel file option
        uploaded_file = st.file_uploader("Upload an Excel file with two columns", type=["xlsx"])

        if uploaded_file is not None:
            try:
                # Read the uploaded Excel file
                df = pd.read_excel(uploaded_file)

                # Display the content of the DataFrame
                st.write("Uploaded Data:")
                st.dataframe(df)

                # Check if the DataFrame has at least two columns
                if df.shape[1] >= 2:
                    sentence1_col = df.columns[0]
                    sentence2_col = df.columns[1]

                    # Calculate similarity when the button is clicked
                    if st.button("Calculate Similarity"):
                        results = []
                        progress_bar = st.progress(0)
                        for i in range(len(df)):
                            sentence1 = df[sentence1_col].iloc[i]
                            sentence2 = df[sentence2_col].iloc[i]

                            # Handle empty sentences
                            if pd.isna(sentence1) or pd.isna(sentence2):
                                similarity_percentage = 0
                            else:
                                # Encode the sentences and calculate similarity
                                embeddings1 = model.encode(sentence1, convert_to_tensor=True)
                                embeddings2 = model.encode(sentence2, convert_to_tensor=True)
                                similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
                                similarity_percentage = similarity.item() * 100

                            # Store results with semantic deviation
                            results.append({
                                "Sentence 1": sentence1,
                                "Sentence 2": sentence2,
                                "Similarity Score": round(similarity.item(), 4),
                                "Similarity Percentage": round(similarity_percentage, 2),
                                "Semantic Deviation": categorize_semantic_deviation(similarity_percentage)
                            })

                            # Update progress bar
                            progress_bar.progress((i+1) / len(df))

                        # Convert results to DataFrame
                        results_df = pd.DataFrame(results)
                        st.write("Similarity Results:")
                        st.dataframe(results_df)

                        # Provide download link for the results
                        excel_data = create_download_link(results_df)
                        st.download_button(
                            label="Download Results as Excel",
                            data=excel_data,
                            file_name="similarity_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error("The uploaded file must contain at least two columns.")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    elif choice == "Manual Input":
        # Input fields for manual sentence typing
        sentence1 = st.text_area("Enter the first sentence:")
        sentence2 = st.text_area("Enter the second sentence:")

        # Calculate similarity when the button is clicked
        if st.button("Calculate Similarity"):
            if sentence1 and sentence2:
                try:
                    # Encode the sentences and calculate similarity
                    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
                    embeddings2 = model.encode(sentence2, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
                    similarity_percentage = similarity.item() * 100

                    # Display the results
                    st.write(f"**Similarity Score:** {similarity.item():.4f}")
                    st.write(f"**Similarity Percentage:** {similarity_percentage:.2f}%")
                    st.write(f"**Semantic Deviation:** {categorize_semantic_deviation(similarity_percentage)}")

                    # Prepare the results as a DataFrame for download
                    result_data = [{
                        "Sentence 1": sentence1,
                        "Sentence 2": sentence2,
                        "Similarity Score": round(similarity.item(), 4),
                        "Similarity Percentage": round(similarity_percentage, 2),
                        "Semantic Deviation": categorize_semantic_deviation(similarity_percentage)
                    }]
                    results_df = pd.DataFrame(result_data)

                    # Provide download link for the results
                    excel_data = create_download_link(results_df)
                    st.download_button(
                        label="Download Result as Excel",
                        data=excel_data,
                        file_name="manual_similarity_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error calculating similarity: {e}")
            else:
                st.error("Please enter both sentences.")

    # Display additional information
    st.markdown("---")
    st.write("### About this App")
    st.write("This app uses the Sentence Transformers library to calculate the semantic similarity between two sentences.")

if __name__ == "__main__":
    main()
