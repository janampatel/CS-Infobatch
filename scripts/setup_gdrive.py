#!/usr/bin/env python3
"""
scripts/setup_gdrive.py
------------------------
One-time setup script to authorize the app with Google Drive and save token.json.

Run this ONCE on your remote server:
    python scripts/setup_gdrive.py

It will print a URL — open it in any browser (on your local machine), complete
the OAuth flow, then paste the authorization code back into the terminal.

After this, all training runs will automatically upload results to Google Drive.

Prerequisites:
    1. Go to https://console.cloud.google.com
    2. Create a project (or use existing)
    3. Enable the Google Drive API
    4. Go to APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID
    5. Application type: Desktop app
    6. Download JSON → rename to credentials.json → put in project root
    7. Run this script
"""

import sys
import os
from pathlib import Path

# Allow running from project root or scripts/ dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"
SCOPES = ['https://www.googleapis.com/auth/drive.file']


def main():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        print(
            "ERROR: Google API libraries not installed.\n"
            "Run:  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
        sys.exit(1)

    if not Path(CREDENTIALS_PATH).exists():
        print(
            f"ERROR: '{CREDENTIALS_PATH}' not found in current directory.\n\n"
            "Steps to get it:\n"
            "  1. Go to https://console.cloud.google.com\n"
            "  2. Select your project → APIs & Services → Credentials\n"
            "  3. Create OAuth 2.0 Client ID (Desktop app)\n"
            "  4. Download JSON and rename it to credentials.json\n"
            "  5. Place credentials.json in the project root\n"
            "  6. Re-run this script"
        )
        sys.exit(1)

    print("=" * 60)
    print("  Google Drive Authorization Setup")
    print("=" * 60)
    print()
    print("A browser window will open. Sign in with your Google")
    print("account and grant the requested permissions.")
    print()

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    # Opens a local browser tab and waits for the redirect
    creds = flow.run_local_server(port=0, open_browser=True)

    # Save token
    with open(TOKEN_PATH, 'w') as f:
        f.write(creds.to_json())

    print()
    print(f"✓ token.json saved successfully.")

    # Test connection
    print()
    print("Testing connection to Google Drive...")
    try:
        service = build('drive', 'v3', credentials=creds)
        about = service.about().get(fields="user").execute()
        user = about.get('user', {})
        print(f"✓ Connected as: {user.get('displayName', 'Unknown')} ({user.get('emailAddress', '?')})")
    except Exception as e:
        print(f"WARNING: Connection test failed: {e}")
        print("token.json was saved — upload may still work during training.")

    print()
    print("=" * 60)
    print("  NEXT STEP: Copy token.json to your remote server")
    print("=" * 60)
    print()
    print("Run this command on your LOCAL machine:")
    print()
    print(f"  scp token.json <user>@<remote_server>:/path/to/treefinder-atml-infobatch/")
    print()
    print("After that, all training runs will auto-upload to Google Drive!")



if __name__ == "__main__":
    main()
