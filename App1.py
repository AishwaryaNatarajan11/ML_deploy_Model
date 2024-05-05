import os
import pickle
import streamlit as st

# Construct the full path to the pickled model file
model_file = os.path.join(os.path.dirname(__file__), 'classifier.pk1')

# Load the pickled model
with open(model_file, 'rb') as pickle_in:
    clf_res = pickle.load(pickle_in)

# Define the Streamlit app
def main():
    st.title('Predictive Maintenance App')
    
    # Add input fields for user to enter feature values
    variance = st.number_input('Variance', min_value=0.0, max_value=100.0)
    skewness = st.number_input('Skewness', min_value=0.0, max_value=100.0)
    curtosis = st.number_input('Curtosis', min_value=0.0, max_value=100.0)
    entropy = st.number_input('Entropy', min_value=0.0, max_value=100.0)
    
    # Add a button to trigger prediction
    if st.button('Predict'):
        # Make prediction using the loaded model
        prediction = clf_res.predict([[variance, skewness, curtosis, entropy]])
        
        # Display the prediction result
        st.success(f'Predicted Failure Type: {prediction[0]}')

# Run the Streamlit app
if __name__ == '__main__':
    main()
    