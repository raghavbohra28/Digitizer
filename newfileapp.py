import requests
import streamlit as st
from streamlit_lottie import st_lottie
import pyfile

import tensorflow as tf
import keras 
from PIL import Image
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
	r=requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()

lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_0fhlytwe.json")

img_thumbnail= Image.open("images/thumbnail.png")
logo= Image.open("images/upeslogo.png")

with st.container():
    left_column,right_column = st.columns([5,1])
    with left_column:
        st.subheader("Welcome to our Minor Project 2 !!")
    with right_column:
        st.image(logo)

with st.container():
	st.title("Interpreting Doctor Notes Using Deep Learning Techniques")
	st.write(" Created By:\t     *Raghav Bohra*,\t    *Rishika Bhalla*,\t     *Avani Vaish*")
	st.write(" We all are pre final year students, doing our BTech in Computer Science with specialization in AI & ML, from UPES, Dehradun.")
	st.write("[UPES Official Website >](https://www.upes.ac.in/)")
	st.write(" This Project was done under the guidance of **Ms. Richa Choudhary** ")

with st.container():
	st.write("---")
	left_column, right_column = st.columns(2)
	with left_column:
		st.header("Objective : ")
		st.write("##")
		st.write("""It can be challenging for the general public and some pharmacists to understand the 
prescriptions that doctors have prescribed because their handwriting is often 
unreadable. 	So we have create a tool that can translate a doctor's handwriting into a 
readable text format using a recognition system.""")

	with right_column:
		st_lottie(lottie_coding, height=300, key="medical")




from tensorflow.keras.layers import Layer

# Define the CTCLayer as a custom layer
class CTCLayer(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

# Load the model with the custom CTCLayer
with tf.keras.utils.custom_object_scope({'CTCLayer': CTCLayer}):
    new_model = tf.keras.models.load_model('C:/Users/shree/0last/handwriting.h5')

def prepare_dataset_custom(img_paths_3):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths_3)).map(
        pyfile.preprocess_image, num_parallel_calls=pyfile.AUTOTUNE
    )
    return dataset.batch(pyfile.batch_size).cache().prefetch(pyfile.AUTOTUNE)

def prediction(image):

    
    custom_ds = prepare_dataset_custom([image])
    
    for batch in custom_ds.take(1):
        MAX_LABEL_LENGTH=21
        batch_images = tf.stack(batch)
        batch_labels = tf.zeros([batch_images.shape[0], MAX_LABEL_LENGTH])
    
        print("len is : ",len(batch))
        print("batch img shape: ",batch_images[0].shape)
    
        _, ax = plt.subplots(1, 1, figsize=(15, 8))

        preds = new_model.predict([batch_images, batch_labels])
        pred_texts = pyfile.decode_batch_predictions(preds)
    
    
        return pred_texts


def read_image_file(uploaded_file):
    # Read the file contents
    contents = uploaded_file.read()

    # Decode the file contents to a tensor with shape (height, width, channels)
    image = tf.image.decode_image(contents)

    # Convert the tensor to a NumPy array and add a channel dimension
    image_array = np.expand_dims(image.numpy(), axis=-1)

    # Resize the array to the expected shape
    image_array = tf.image.resize(image_array, (128, 32)).numpy()

    return image_array

# Define the Streamlit app
def main():
    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image of the handwritten word for prediction : ", type=['jpg', 'jpeg', 'png'])
    
    # If the user uploaded a file
    if uploaded_file is not None:
        # Read the image file
        #image = read_image_file(uploaded_file)
        image=uploaded_file.name
        # Make a prediction with the model
        final_answer = prediction(image)
        
        st.image(uploaded_file, caption="Uploaded image")
        st.write("Prediction: ",final_answer)

if __name__ == '__main__':
    main()
	
with st.container():
    st.write("---")
    st.header("Presentation")
    st.write("##")
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(img_thumbnail)
    with text_column:
        st.subheader("Interpreting Doctor Notes Using Deep Learning Techniques")
        st.write(
            """
            Minor Project 2 Final Presentation\n
            Have a look on it!!!
  
            """
        )
        st.markdown("[Go to the Presentation...](https://docs.google.com/presentation/d/1O-IWXfRZxYg2eTxJ0ipG2TxXsPH09qv049vfjpQpKiI/edit?usp=sharing)")

with st.container():
	st.write("---")
	st.write("<h1 style='text-align: center;'>Thank You For Visiting...</h1>", unsafe_allow_html=True)



