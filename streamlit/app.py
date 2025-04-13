import streamlit as st
import pandas as pd
from PIL import Image
import io
import tempfile
import requests  # Using requests instead of gdown.download
import hashlib
from datetime import datetime
import os

# Import Supabase client
from supabase import create_client, Client

# Import Google Drive API libraries
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ---------------------
# App Configuration
# ---------------------
st.set_page_config(page_title="People Counter App")
st.title("People Counter App")
st.write("Code images from a Google Drive folder.")

# ---------------------
# Supabase Client Setup
# ---------------------
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = get_supabase_client()

# ---------------------
# Google Drive API Client Setup
# ---------------------
@st.cache_resource
def get_drive_service():
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    credentials_info = st.secrets["gdrive"]["service_account"]
    credentials = service_account.Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=credentials)

drive_service = get_drive_service()

# ---------------------
# Helper Functions
# ---------------------
def get_image_hash(image):
    """Generate a unique hash for an image."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

def get_data_from_supabase():
    """Fetch data from the Supabase 'people_counts' table."""
    try:
        response = supabase.table("people_counts").select("*").execute()
        data = response.data
        if data is None:
            return pd.DataFrame(columns=["image", "image_hash", "women", "men", "timestamp"])
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching data from Supabase: {e}")
        return pd.DataFrame(columns=["image", "image_hash", "women", "men", "timestamp"])

def save_data_to_supabase(num_women, num_men, image_info):
    """Insert or update coded counts in Supabase."""
    try:
        image_hash = image_info["hash"]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = supabase.table("people_counts").select("*").eq("image_hash", image_hash).execute()
        if response.data:
            supabase.table("people_counts").update({
                "women": num_women,
                "men": num_men,
                "timestamp": now_str
            }).eq("image_hash", image_hash).execute()
        else:
            record = {
                "image": image_info["filename"],
                "image_hash": image_hash,
                "women": num_women,
                "men": num_men,
                "timestamp": now_str
            }
            supabase.table("people_counts").insert(record).execute()
        return True
    except Exception as e:
        st.error(f"Error saving data to Supabase: {e}")
        return False

def extract_folder_id(link):
    """Extract the folder ID from a Google Drive folder link."""
    try:
        # Assumes a link like "https://drive.google.com/drive/folders/{folder_id}?..."
        folder_id = link.split("folders/")[1].split("?")[0]
        return folder_id
    except Exception:
        return None

def list_image_files(folder_id):
    """List image files (IDs and names) in a Google Drive folder."""
    query = f"'{folder_id}' in parents and trashed=false and (mimeType contains 'image/')"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return results.get('files', [])

# ---------------------
# Session State Initialization
# ---------------------
if "gdrive_file_list" not in st.session_state:
    st.session_state.gdrive_file_list = None
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# ---------------------
# Folder Link Input and File List Loading
# ---------------------
st.markdown("### Enter Google Drive Folder Link")
folder_link = st.text_input("Folder link (accessible to anyone with the link)")

if folder_link and st.button("Load Folder"):
    folder_id = extract_folder_id(folder_link)
    if not folder_id:
        st.error("Could not extract folder ID. Please check the link format.")
    else:
        files = list_image_files(folder_id)
        if not files:
            st.error("No image files found in the folder.")
        else:
            st.session_state.gdrive_file_list = files
            st.session_state.current_index = 0
            st.success(f"Loaded {len(files)} files from the folder.")

# ---------------------
# Lazy Loading: Find and Display Next Uncoded Image
# ---------------------
image_info = None
df = get_data_from_supabase()
coded_hashes = set(df["image_hash"].tolist()) if not df.empty else set()

if st.session_state.gdrive_file_list:
    files = st.session_state.gdrive_file_list
    idx = st.session_state.current_index
    while idx < len(files):
        file_item = files[idx]
        file_id = file_item["id"]
        filename = file_item["name"]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_filename = temp_file.name
        temp_file.close()
        try:
            response = requests.get(download_url)
            response.raise_for_status()  # Raise an error for bad status codes
            with open(temp_filename, "wb") as f:
                f.write(response.content)
            img = Image.open(temp_filename)
        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
            idx += 1
            continue
        hash_val = get_image_hash(img)
        if hash_val in coded_hashes:
            idx += 1  # Skip already coded image
            continue
        else:
            image_info = {"path": temp_filename, "filename": filename, "hash": hash_val}
            st.session_state.current_index = idx
            break
    if idx >= len(files) and image_info is None:
        st.info("All images in this folder have been coded.")

# ---------------------
# Coding Interface for the Current Image
# ---------------------
if image_info:
    try:
        current_img = Image.open(image_info["path"])
        st.image(current_img, caption=f"Image: {image_info['filename']}", use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

    col1, col2 = st.columns(2)
    with col1:
        num_women = st.number_input("Number of Women", min_value=0, value=0, step=1, key="women")
    with col2:
        num_men = st.number_input("Number of Men", min_value=0, value=0, step=1, key="men")
    
    if st.button("Save Counts for This Image"):
        if save_data_to_supabase(num_women, num_men, image_info):
            st.success(f"Saved counts for {image_info['filename']}")
            st.session_state.current_index += 1
            st.experimental_rerun()

# ---------------------
# Display Coded Data and CSV Download
# ---------------------
df = get_data_from_supabase()
if not df.empty:
    st.write("### Coded Data")
    st.dataframe(df[["image", "women", "men", "timestamp"]])
    csv_data = df.to_csv(index=False)
    st.download_button("Download CSV", csv_data, "people_counts.csv", "text/csv")
