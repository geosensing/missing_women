import streamlit as st
import pandas as pd
from PIL import Image
import io
import tempfile
import gdown
import hashlib
from datetime import datetime
import os

# Import the Supabase client
from supabase import create_client, Client

# Set page title
st.set_page_config(page_title="People Counter App")

st.title("People Counter App")
st.write("Count the number of men and women in images from Google Drive")

@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = get_supabase_client()

# Function to generate a unique hash for an image
def get_image_hash(image):
    """Generate a hash from image data to uniquely identify it"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

# Function to fetch data from Supabase
def get_data_from_supabase():
    """Fetch data from Supabase table 'people_counts'."""
    try:
        response = supabase.table("people_counts").select("*").execute()
        data = response.data
        if data is None:
            return pd.DataFrame(columns=["image", "image_hash", "women", "men", "timestamp"])
        else:
            return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching data from Supabase: {e}")
        return pd.DataFrame(columns=["image", "image_hash", "women", "men", "timestamp"])

# Function to save (insert or update) data in Supabase
def save_data_to_supabase(num_women, num_men):
    if 'current_image' not in st.session_state or st.session_state.current_image is None:
        st.error("No image loaded.")
        return False
    try:
        existing_response = supabase.table("people_counts").select("*").eq("image_hash", st.session_state.current_image_hash).execute()
        existing_records = existing_response.data
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if existing_records:
            supabase.table("people_counts").update({
                "women": num_women,
                "men": num_men,
                "timestamp": now_str
            }).eq("image_hash", st.session_state.current_image_hash).execute()
        else:
            new_record = {
                "image": st.session_state.current_image_name,
                "image_hash": st.session_state.current_image_hash,
                "women": num_women,
                "men": num_men,
                "timestamp": now_str
            }
            supabase.table("people_counts").insert(new_record).execute()
        return True
    except Exception as e:
        st.error(f"Error saving data to Supabase: {e}")
        return False

# Initialize session state variables
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None
if "current_image_hash" not in st.session_state:
    st.session_state.current_image_hash = None

# Google Drive Link Input
st.markdown("### Enter Google Drive Image Link")
drive_link = st.text_input("Enter Google Drive sharing link (must be accessible to anyone with the link)")

if drive_link and st.button("Load Image"):
    try:
        # Extract the file ID from the Google Drive link
        file_id = None
        if "drive.google.com/file/d/" in drive_link:
            file_id = drive_link.split("/file/d/")[1].split("/")[0]
        elif "drive.google.com/open?id=" in drive_link:
            file_id = drive_link.split("id=")[1]
        
        if file_id:
            # Create a temporary file to store the downloaded image
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_filename = temp_file.name
            temp_file.close()
            
            # Download the image using gdown
            download_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(download_url, temp_filename, quiet=False)
            
            # Open and display the image
            image = Image.open(temp_filename)
            image_hash = get_image_hash(image)
            
            # Check if the image has already been coded
            df = get_data_from_supabase()
            if not df.empty and image_hash in df["image_hash"].values:
                existing_record = df[df["image_hash"] == image_hash].iloc[0]
                st.warning("⚠️ This image has already been coded! Previous data shown below.")
                st.info(f"Women: {existing_record['women']}, Men: {existing_record['men']}")
            
            # Save in session state
            st.session_state.current_image = image
            st.session_state.current_image_name = f"drive_file_{file_id}"
            st.session_state.current_image_hash = image_hash
            
            st.success("Image loaded successfully!")
        else:
            st.error("Could not extract file ID from the provided link.")
    except Exception as e:
        st.error(f"Error loading image from Google Drive: {e}")

# Main counting interface
if st.session_state.current_image:
    st.image(st.session_state.current_image,
             caption=f"Image: {st.session_state.current_image_name}",
             use_column_width=True)
    
    # Create two columns for counters
    col1, col2 = st.columns(2)
    
    default_women = 0
    default_men = 0
    df = get_data_from_supabase()
    if not df.empty and st.session_state.current_image_hash in df["image_hash"].values:
        existing_record = df[df["image_hash"] == st.session_state.current_image_hash].iloc[0]
        default_women = existing_record["women"]
        default_men = existing_record["men"]
    
    with col1:
        num_women = st.number_input("Number of Women", min_value=0, value=default_women, step=1)
    with col2:
        num_men = st.number_input("Number of Men", min_value=0, value=default_men, step=1)
    
    if st.button("Save Counts"):
        if save_data_to_supabase(num_women, num_men):
            st.success(f"Saved counts for {st.session_state.current_image_name}")
        else:
            st.error("Failed to save data")

df = get_data_from_supabase()
if not df.empty:
    st.write("### Saved Counts")
    display_df = df[["image", "women", "men", "timestamp"]].copy()
    st.dataframe(display_df)
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="people_counts.csv",
        mime="text/csv"
    )
