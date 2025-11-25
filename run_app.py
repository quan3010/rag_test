import os
import subprocess
from pathlib import Path

import click


@click.command()
@click.option(
    "--mongo-username",
    prompt="Enter your MongoDB username",
    help="MongoDB username (e.g., om1325)",
)
@click.option(
    "--mongo-password",
    prompt="Enter your MongoDB password",
    hide_input=True,
    confirmation_prompt=False,
    help="MongoDB password",
)
@click.option(
    "--mongo-host",
    prompt="Enter your MongoDB host",
    default="cluster0.lqirl.mongodb.net",
    show_default=True,
    help="MongoDB host (without protocol)",
)
def main(mongo_username, mongo_password, mongo_host):
    """Prompt for MongoDB creds and launch the Streamlit app."""
    # Build the URI
    mongo_uri = (
        f"mongodb+srv://{mongo_username}:{mongo_password}"
        f"@{mongo_host}/?retryWrites=true&w=majority&appName=Cluster0"
    )
    print(mongo_uri)
    # Export for the Streamlit app
    os.environ["MONGO_URI"] = mongo_uri

    # Optionally: print a small confirmation (but not the password!)
    click.echo("✅ MongoDB URI configured. Launching Streamlit app...")

    app_path = Path(__file__).parent / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
