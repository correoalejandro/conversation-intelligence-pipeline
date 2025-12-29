#!/usr/bin/env python3
"""
Dynamic CLI Menu: Orchestrates CLI reports, SAS upload, Web UI, and more
"""

import os
import subprocess
from _data_layer.api import load_artifact
from _cli_helpers.input_selector import select_artifact

# Menu action registry
MENU_ACTIONS = {}

def register_action(label):
    """Decorator to register a menu action"""
    def decorator(func):
        MENU_ACTIONS[str(len(MENU_ACTIONS) + 1)] = (label, func)
        return func
    return decorator

# === Actions ===

@register_action("🖥 CLI Report Demo")
def cli_report_demo():
    print("\n📂 Please select a Data Layer artifact:")
    artifact_id = select_artifact(stage="processPipeline")
    df, meta = load_artifact(artifact_id)
    from _presentation_layer.cli_reporter import print_experiment_report
    print("\n📊 Running CLI Report Demo...")
    print_experiment_report(df, meta)

@register_action("🌐 Web UI Demo (Streamlit)")
def web_ui_demo():
    print("\n📂 Please select a Data Layer artifact:")
    artifact_id = select_artifact(stage="processPipeline")
    os.environ["ARTIFACT_ID"] = artifact_id
    print("\n🌐 Launching Web UI...")
    subprocess.Popen(["streamlit", "run", "4.v6.web_ui_app.py"], shell=True)
    print("✅ Streamlit is running at http://localhost:8501")

@register_action("📤 SAS Upload Demo")
def sas_upload_demo():
    print("\n📂 Please select a Data Layer artifact:")
    artifact_id = select_artifact(stage="processPipeline")
    df, meta = load_artifact(artifact_id)
    table_name = input("Enter CAS table name for upload: ").strip()
    from _presentation_layer.sas_exporter import upload_dataframe_to_sas
    upload_dataframe_to_sas(df, table_name=table_name, caslib="Public", promote=True)

@register_action("📚 Artifact Registry Explorer")
def artifact_registry_explorer():
    from _presentation_layer.cli_reporter import print_registry_summary
    print("\n📚 Launching Artifact Registry Explorer...")
    print_registry_summary()

@register_action("❌ Exit")
def exit_menu():
    print("👋 Goodbye!")
    raise SystemExit

# === Menu Loop ===

def menu():
    """Dynamic interactive CLI menu"""
    while True:
        print("\n=== Presentation Layer Menu ===")
        for key, (label, _) in MENU_ACTIONS.items():
            print(f"{key}. {label}")

        choice = input(f"Select an option [1-{len(MENU_ACTIONS)}]: ").strip()
        action = MENU_ACTIONS.get(choice)

        if action:
            try:
                action[1]()
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    menu()
