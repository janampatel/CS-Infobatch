"""Upload experiment outputs (results, logs, checkpoints) to Google Drive via
the Drive API v3 with OAuth 2.0. One-time setup: download credentials.json
from Google Cloud Console, then run python scripts/setup_gdrive.py."""

import os
import logging
import mimetypes
from pathlib import Path

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.file']


def get_gdrive_service(credentials_path: str, token_path: str):
    """Authenticate and return a Drive API service object; uses saved token or OAuth flow."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "Google API libraries not installed. Run:\n"
            "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )

    creds = None

    if Path(token_path).exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("[GDrive] Refreshing expired token...")
            creds.refresh(Request())
        else:
            if not Path(credentials_path).exists():
                raise FileNotFoundError(
                    f"[GDrive] credentials.json not found at '{credentials_path}'.\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials.\n"
                    "Then run: python scripts/setup_gdrive.py"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES,
                redirect_uri="urn:ietf:wg:oauth:2.0:oob"
            )
            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f"\n[GDrive] Open this URL in your browser:\n\n  {auth_url}\n")
            code = input("[GDrive] Paste the authorization code: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials

        with open(token_path, 'w') as f:
            f.write(creds.to_json())
        logger.info(f"[GDrive] Token saved to {token_path}")

    service = build('drive', 'v3', credentials=creds)
    return service


def get_or_create_folder(service, name: str, parent_id: str = None) -> str:
    """Get the ID of a Drive folder by name under parent_id, creating it if needed."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    if files:
        return files[0]['id']

    metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder',
    }
    if parent_id:
        metadata['parents'] = [parent_id]

    folder = service.files().create(body=metadata, fields='id').execute()
    logger.info(f"[GDrive] Created folder '{name}' (id={folder['id']})")
    return folder['id']


def upload_file(service, local_path: Path, parent_folder_id: str) -> str:
    """Upload a file to Drive under parent_folder_id; updates in place if it already exists."""
    from googleapiclient.http import MediaFileUpload

    name = local_path.name
    mime_type, _ = mimetypes.guess_type(str(local_path))
    mime_type = mime_type or 'application/octet-stream'

    query = f"name='{name}' and '{parent_folder_id}' in parents and trashed=false"
    existing = service.files().list(q=query, fields="files(id)").execute().get('files', [])

    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)

    if existing:
        file_id = existing[0]['id']
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        metadata = {'name': name, 'parents': [parent_folder_id]}
        result = service.files().create(body=metadata, media_body=media, fields='id').execute()
        file_id = result['id']

    return file_id


def upload_folder_recursive(service, local_dir: Path, parent_folder_id: str):
    """Recursively upload local_dir to Drive under parent_folder_id, mirroring structure."""
    for item in sorted(local_dir.iterdir()):
        if item.is_dir():
            sub_folder_id = get_or_create_folder(service, item.name, parent_folder_id)
            upload_folder_recursive(service, item, sub_folder_id)
        elif item.is_file():
            upload_file(service, item, parent_folder_id)
            logger.info(f"[GDrive] Uploaded: {item}")


def upload_experiment(exp_name: str, cfg: dict, gdrive_cfg: dict):
    """Upload results/<exp_name>/, checkpoints/<exp_name>/, and logs/<exp_name>*.log to Drive."""
    credentials_path = gdrive_cfg.get('credentials_path', 'credentials.json')
    token_path = gdrive_cfg.get('token_path', 'token.json')
    drive_root_name = gdrive_cfg.get('drive_folder', 'treefinder-atml-runs')

    logger.info(f"[GDrive] Starting upload for experiment: {exp_name}")

    try:
        service = get_gdrive_service(credentials_path, token_path)
    except Exception as e:
        logger.warning(f"[GDrive] Authentication failed — skipping upload. Error: {e}")
        return

    try:
        root_id = get_or_create_folder(service, drive_root_name)
        exp_id = get_or_create_folder(service, exp_name, root_id)

        results_dir = Path(cfg['output']['results_dir']) / exp_name
        if results_dir.exists() and any(results_dir.iterdir()):
            results_folder_id = get_or_create_folder(service, 'results', exp_id)
            upload_folder_recursive(service, results_dir, results_folder_id)
            logger.info(f"[GDrive] results/ uploaded.")
        else:
            logger.info(f"[GDrive] No results dir found at {results_dir}, skipping.")

        ckpt_dir = Path(cfg['output']['checkpoint_dir']) / exp_name
        if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
            ckpt_folder_id = get_or_create_folder(service, 'checkpoints', exp_id)
            upload_folder_recursive(service, ckpt_dir, ckpt_folder_id)
            logger.info(f"[GDrive] checkpoints/ uploaded.")
        else:
            logger.info(f"[GDrive] No checkpoint dir found at {ckpt_dir}, skipping.")

        log_dir = Path(cfg['logging']['log_dir'])
        if log_dir.exists():
            log_files = list(log_dir.glob(f"{exp_name}*"))
            if log_files:
                logs_folder_id = get_or_create_folder(service, 'logs', exp_id)
                for log_file in log_files:
                    if log_file.is_file():
                        upload_file(service, log_file, logs_folder_id)
                        logger.info(f"[GDrive] Uploaded log: {log_file}")
            else:
                logger.info(f"[GDrive] No log files found for {exp_name} in {log_dir}.")

        logger.info(
            f"[GDrive] Upload complete! Find your files at:\n"
            f"  https://drive.google.com  →  '{drive_root_name}/{exp_name}/'"
        )

    except Exception as e:
        logger.warning(
            f"[GDrive] Upload failed — your results are still saved locally. Error: {e}"
        )
