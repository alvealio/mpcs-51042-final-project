import requests
import json
from urllib.parse import urlencode

# https://developers.strava.com/docs/reference/#api-Uploads-createUpload
# A lot of dependencies associated with Strava's standard API usage. Just use requests instead

# Strava API Credentials
with open("config.json", "r") as f:
    config = json.load(f)

CLIENT_ID = config["STRAVA_CLIENT_ID"]
CLIENT_SECRET = config["STRAVA_CLIENT_SECRET"]
REDIRECT_URI = "http://127.0.0.1:5000/strava_callback"
AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
UPLOAD_URL = "https://www.strava.com/api/v3/uploads"


# Authentication guide: https://developers.strava.com/docs/authentication/
def get_auth_url():
    """
    Return Strava OAuth URL

    Returns:
        str: URL to redirect user for Strava authentication.
    """
    params = {
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'approval_prompt': 'auto',
        'scope': 'activity:write'
    }
    return f"{AUTH_URL}?{urlencode(params)}"

def get_access_token(auth_code):
    """
    Get access token with authorization code

    Args:
        auth_code (str): Authorization code returned by Strava.
    
    Returns:
        dict: A dictionary containing access token and ancillary information.
    """
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': auth_code,
        'grant_type': 'authorization_code'
    }
    response = requests.post(TOKEN_URL, data=payload)
    response.raise_for_status
    return response.json()


# Uploading to Strava: https://developers.strava.com/docs/uploads/
def upload_gpx(file, access_token, activity_name="Strava Art", data_type="gpx"):
    """
    Uploads GPX file to Strava account.

    Args:
        file (str): File path of the GPX file.
        access_token (str): OAuth access token.
        activity_name (str, optional): Desired name of uploaded activity.
        data_type (str, optional): Format of uploaded file (GPX is default)
    
    Returns:
        dict: JSON response from Strava API
    """
    header = {
        "Authorization": f"Bearer {access_token}"
    }
    file = {
        "file": open(file, "rb")
    }
    payload = {
        "data_type": data_type,
        "name": activity_name
    }
    response = requests.post(UPLOAD_URL, headers=header, files=file, data=payload)
    response.raise_for_status()
    return response.json()

