#!/usr/bin/env python3
"""One-time OAuth setup for Google Drive uploads; saves token.json.

Run once on the remote server: python scripts/setup_gdrive.py
Prereqs: create a Google Cloud project, enable the Drive API, create an
OAuth 2.0 Desktop app credential, download it as credentials.json in the
project root, then run this script and follow the printed auth URL.
"""

import sys
import os
from pathlib import Path

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
