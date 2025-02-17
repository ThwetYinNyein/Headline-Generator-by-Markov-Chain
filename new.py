import streamlit as st
import markovify
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO

# Function to train a new Markov chain model from user-uploaded text
def train_new_model(text_data, state_size=2):
    text_model = markovify.Text(text_data, state_size=state_size)
    return text_model

# Function to generate headlines using the trained Markov chain model
def generate_headlines(text_model, num_headlines=5):
    headlines = []
    for _ in range(num_headlines):
        headline = text_model.make_sentence()
        if headline:
            headlines.append(headline)
    return headlines

# Function to visualize the Markov chain
def visualize_markov_chain(model):
    transition_matrix = model.chain.model
    G = nx.DiGraph()

    for current_state, next_states in transition_matrix.items():
        for next_state, count in next_states.items():
            G.add_edge(current_state, next_state, weight=count)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    st.pyplot(plt)

# Main function to run the Streamlit app
def main():
    st.title("Markov Chain Headline Generator")
    st.image("banner.png", use_container_width=True)
    st.write("This app generates random headlines using a Markov chain model trained on your own text data.")

    # Sidebar for user options
    st.sidebar.header("Options")
    upload_text = st.sidebar.file_uploader("Upload your own text file to train a Markov model", type=["txt", "csv"])

    if upload_text is not None:
        if upload_text.type == "text/csv":
            df = pd.read_csv(upload_text)
            text_data = "\n".join(df.iloc[:, 0])
        else:
            text_data = upload_text.getvalue().decode("utf-8")
        
        state_size = st.sidebar.slider("Select State Size for Markov Chain", 1, 5, 2)
        text_model = train_new_model(text_data, state_size=state_size)
        st.write("Markov model trained successfully!")

        num_headlines = st.slider("Number of headlines to generate", 1, 20, 5)
        
        if st.button("Generate Headlines"):
            st.write("Generated Headlines:")
            headlines = generate_headlines(text_model, num_headlines)
            for i, headline in enumerate(headlines, 1):
                st.write(f"{i}. {headline}")

            export_file = StringIO()
            export_file.write("\n".join(headlines))
            st.download_button(
                label="Download Headlines as Text File",
                data=export_file.getvalue(),
                file_name="generated_headlines.txt",
                mime="text/plain"
            )
    else:
        st.warning("Please upload a text file to train the Markov model.")

if __name__ == "__main__":
    main()
