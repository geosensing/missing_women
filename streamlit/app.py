import streamlit as st
import pandas as pd
from PIL import Image
import io
import tempfile
import gdown
import hashlib
from datetime import datetime
import os
import json
import base64
from github import Github

# Set page title
st.set_page_config(page_title="People Counter App")

st.title("People Counter App")
st.write("Count the number of men and women in images from Google Drive")

# Function to initialize GitHub connection
@st.cache_resource
def get_github_connection():
    """Initialize GitHub connection if not already done"""
    try:
        # Check if we have the access token in Streamlit secrets
        if 'github' in st.secrets and 'access_token' in st.secrets['github']:
            g = Github(st.secrets['github']['access_token'])
            return g
        else:
            # For local development, use a token from the environment
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                g = Github(github_token)
                return g
            else:
                st.error("GitHub token not found in secrets or environment")
                return None
    except Exception as e:
        st.error(f"Failed to initialize GitHub connection: {e}")
        return None

# Initialize GitHub
github_client = get_github_connection()

# Function to get the repository
@st.cache_resource
def get_github_repo():
    """Get the GitHub repository"""
    if not github_client:
        return None
    
    try:
        # Get repository from secrets
        if 'github' in st.secrets and 'repo' in st.secrets['github']:
            repo_name = st.secrets['github']['repo']
            return github_client.get_repo(repo_name)
        else:
            # For local development
            repo_name = os.environ.get('GITHUB_REPO')
            if repo_name:
                return github_client.get_repo(repo_name)
            else:
                st.error("GitHub repository not found in secrets or environment")
                return None
    except Exception as e:
        st.error(f"Failed to get GitHub repository: {e}")
        return None

# Get repository
repo = get_github_repo()

# Function to generate a unique hash for an image
def get_image_hash(image):
    """Generate a hash from image data to uniquely identify it"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

# Function to read data from CSV file in GitHub repo
def get_data_from_github():
    """Fetch data from CSV file in GitHub repository"""
    if not repo:
        return pd.DataFrame(columns=["Image", "ImageHash", "Women", "Men", "Timestamp"])
    
    try:
        # Try to get the data file from the repo
        file_path = 'data/people_counts.csv'
        try:
            contents = repo.get_contents(file_path)
            # Decode content
            content = contents.decoded_content.decode('utf-8')
            # Convert to DataFrame
            df = pd.read_csv(io.StringIO(content))
            return df
        except Exception as e:
            # File doesn't exist yet, return empty DataFrame
            return pd.DataFrame(columns=["Image", "ImageHash", "Women", "Men", "Timestamp"])
    except Exception as e:
        st.error(f"Error fetching data from GitHub: {e}")
        return pd.DataFrame(columns=["Image", "ImageHash", "Women", "Men", "Timestamp"])

# Function to save data to GitHub
def save_data_to_github(df):
    """Save DataFrame to CSV file in GitHub repository"""
    if not repo:
        st.error("GitHub repository not initialized")
        return False
    
    try:
        # Convert DataFrame to CSV
        csv_content = df.to_csv(index=False)
        
        # File path in the repository
        file_path = 'data/people_counts.csv'
        
        # Check if the file already exists
        try:
            # Try to get the file to check if it exists
            contents = repo.get_contents(file_path)
            
            # Update the file
            repo.update_file(
                path=file_path,
                message="Update people count data",
                content=csv_content,
                sha=contents.sha
            )
        except Exception as e:
            # If the file doesn't exist, create it
            try:
                # Check if the data directory exists
                try:
                    repo.get_contents('data')
                except:
                    # Create data directory
                    repo.create_file(
                        path='data/.gitkeep',
                        message="Create data directory",
                        content=""
                    )
                
                # Create the file
                repo.create_file(
                    path=file_path,
                    message="Create people count data file",
                    content=csv_content
                )
            except Exception as create_error:
                st.error(f"Error creating file: {create_error}")
                return False
        
        return True
    except Exception as e:
        st.error(f"Error saving data to GitHub: {e}")
        return False

# Initialize session state
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_image_name' not in st.session_state:
    st.session_state.current_image_name = None
if 'current_image_hash' not in st.session_state:
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
            
            # Download the file using gdown
            download_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(download_url, temp_filename, quiet=False)
            
            # Open and display the image
            image = Image.open(temp_filename)
            
            # Generate hash for image
            image_hash = get_image_hash(image)
            
            # Check if this image has already been coded
            df = get_data_from_github()
            if not df.empty and image_hash in df['ImageHash'].values:
                existing_record = df[df['ImageHash'] == image_hash].iloc[0]
                st.warning("⚠️ This image has already been coded! Previous data shown below.")
                st.info(f"Women: {existing_record['Women']}, Men: {existing_record['Men']}")
            
            # Store in session state
            st.session_state.current_image = image
            st.session_state.current_image_name = f"drive_file_{file_id}"
            st.session_state.current_image_hash = image_hash
            
            st.success("Image loaded successfully!")
        else:
            st.error("Could not extract file ID from the provided link.")
            
    except Exception as e:
        st.error(f"Error loading image from Google Drive: {str(e)}")

# Fallback: Direct upload option
st.markdown("### Or upload an image from your device")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the image
    image = Image.open(uploaded_file)
    
    # Generate hash for image
    image_hash = get_image_hash(image)
    
    # Check if this image has already been coded
    df = get_data_from_github()
    if not df.empty and image_hash in df['ImageHash'].values:
        existing_record = df[df['ImageHash'] == image_hash].iloc[0]
        st.warning("⚠️ This image has already been coded! Previous data shown below.")
        st.info(f"Women: {existing_record['Women']}, Men: {existing_record['Men']}")
    
    # Store in session state
    st.session_state.current_image = image
    st.session_state.current_image_name = uploaded_file.name
    st.session_state.current_image_hash = image_hash
    
    st.success("Image uploaded successfully!")

# Display the current image and counting interface
if st.session_state.current_image:
    st.image(st.session_state.current_image, caption=f"Image: {st.session_state.current_image_name}", use_column_width=True)
    
    # Create columns for counters
    col1, col2 = st.columns(2)
    
    # Determine default values (use previous values if they exist)
    default_women = 0
    default_men = 0
    
    # Get data and check if this image has already been coded
    df = get_data_from_github()
    if not df.empty and st.session_state.current_image_hash in df['ImageHash'].values:
        existing_record = df[df['ImageHash'] == st.session_state.current_image_hash].iloc[0]
        default_women = existing_record['Women']
        default_men = existing_record['Men']
    
    with col1:
        num_women = st.number_input("Number of women", min_value=0, value=default_women, step=1)
    
    with col2:
        num_men = st.number_input("Number of men", min_value=0, value=default_men, step=1)
    
    # Save button
    if st.button("Save Counts"):
        # Get current data
        df = get_data_from_github()
        
        # Check if this image has already been coded
        if not df.empty and st.session_state.current_image_hash in df['ImageHash'].values:
            # Update existing record
            df.loc[df['ImageHash'] == st.session_state.current_image_hash, 'Women'] = num_women
            df.loc[df['ImageHash'] == st.session_state.current_image_hash, 'Men'] = num_men
            df.loc[df['ImageHash'] == st.session_state.current_image_hash, 'Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Create new record
            new_row = {
                "Image": st.session_state.current_image_name,
                "ImageHash": st.session_state.current_image_hash,
                "Women": num_women,
                "Men": num_men,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            # Add to dataframe
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to GitHub
        if save_data_to_github(df):
            st.success(f"Saved counts for {st.session_state.current_image_name}")
        else:
            st.error("Failed to save data")

# Display the saved data
df = get_data_from_github()
if not df.empty:
    st.write("### Saved Counts")
    
    # Create a display version without the hash (for cleaner UI)
    display_df = df[['Image', 'Women', 'Men', 'Timestamp']].copy()
    st.dataframe(display_df)
    
    # Download button for the data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="people_counts.csv",
        mime="text/csv"
    )

# Add instructions about GitHub setup
with st.expander("📝 Setup Instructions"):
    st.markdown("""
    ### GitHub Setup Instructions
    
    To use this app with GitHub storage, you need to:
    
    1. **Create a GitHub Personal Access Token**:
       - Go to your GitHub account settings
       - Click on "Developer settings" > "Personal access tokens" > "Tokens (classic)"
       - Generate a new token with repo scope permissions
    
    2. **Set Up Secrets in Streamlit Cloud**:
       - In your GitHub repository, create `.streamlit/secrets.toml`
       - Add your GitHub credentials in this format:
    
    ```toml
    [github]
    access_token = "your-github-personal-access-token"
    repo = "your-username/your-repository-name"
    ```
    
    3. **Required Packages**:
    ```
    streamlit==1.27.0
    pandas==2.0.3
    pillow==10.0.0
    gdown==4.7.1
    PyGithub==1.58.2
    ```
    
    The app will store all data in a CSV file at `data/people_counts.csv` in your GitHub repository.
    """)