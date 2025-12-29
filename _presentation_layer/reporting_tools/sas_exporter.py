# _presentation_layer/sas_exporter.py

"""
SAS Exporter
------------
Handles formatting and uploading DataFrames to SAS CAS servers.
Uses _connections_layer.sas_helper for connection management.
"""

from _connections_layer.sas_helper import httpConnSwat

def upload_dataframe_to_sas(df, table_name, caslib="Public", promote=True):
    """
    Uploads a DataFrame to SAS CAS using the SAS helper functions.

    Args:
        df (pandas.DataFrame): DataFrame to upload.
        table_name (str): Name of the target CAS table.
        caslib (str): CAS library to upload into (default: 'Public').
        promote (bool): Whether to promote the table in CAS.
    """
    # Get SAS CAS connection
    conn = httpConnSwat()

    print(f"📤 Uploading DataFrame to CAS (caslib='{caslib}', table='{table_name}')...")
    conn.upload_frame(df, caslib=caslib, table_name=table_name, promote=promote)
    print("✅ Upload successful.")

    # Optionally terminate session
    conn.terminate()
    print("🔒 CAS session terminated.")
