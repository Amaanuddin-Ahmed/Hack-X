"""Download the research model checkpoint into the deployment bundle."""

from service import load_model


if __name__ == "__main__":
    load_model()
    print("Chest X-ray model checkpoint is ready.")
