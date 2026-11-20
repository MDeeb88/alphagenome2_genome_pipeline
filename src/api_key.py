#!/usr/bin/env python3

import getpass
import os
from pathlib import Path


ALPHAGENOME_DOCS_URL = (
    "https://www.alphagenomedocs.com/"
)

API_KEY_DIR = (
    Path.home()
    / ".config"
    / "alphagenome2"
)

API_KEY_FILE = (
    API_KEY_DIR
    / "api_key"
)


def read_saved_api_key():
    if not API_KEY_FILE.exists():
        return None

    try:
        key = (
            API_KEY_FILE
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        return key or None

    except OSError:
        return None


def load_api_key():
    """
    Load the API key into ALPHA_GENOME_API_KEY.

    Priority:
      1. Existing environment variable
      2. Saved per-user key file
    """

    existing = (
        os.environ.get(
            "ALPHA_GENOME_API_KEY",
            "",
        )
        .strip()
    )

    if existing:
        return True

    saved = read_saved_api_key()

    if saved:
        os.environ[
            "ALPHA_GENOME_API_KEY"
        ] = saved

        return True

    return False


def api_key_status():
    env_key = (
        os.environ.get(
            "ALPHA_GENOME_API_KEY",
            "",
        )
        .strip()
    )

    saved_key = (
        read_saved_api_key()
    )

    if env_key:
        if (
            saved_key
            and env_key == saved_key
        ):
            return (
                "CONFIGURED "
                "(saved for this Linux user)"
            )

        return (
            "CONFIGURED "
            "(current session/environment)"
        )

    if saved_key:
        return (
            "CONFIGURED "
            "(saved, not yet loaded)"
        )

    return "NOT CONFIGURED"


def save_api_key(key):
    key = key.strip()

    if not key:
        raise ValueError(
            "API key cannot be empty."
        )

    API_KEY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # User-only access to the directory.
    #
    os.chmod(
        API_KEY_DIR,
        0o700,
    )

    temp_path = (
        API_KEY_DIR
        / "api_key.tmp"
    )

    fd = os.open(
        temp_path,
        (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_TRUNC
        ),
        0o600,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                key + "\n"
            )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise

    os.replace(
        temp_path,
        API_KEY_FILE,
    )

    os.chmod(
        API_KEY_FILE,
        0o600,
    )

    os.environ[
        "ALPHA_GENOME_API_KEY"
    ] = key


def remove_saved_api_key():
    saved = read_saved_api_key()

    if API_KEY_FILE.exists():
        API_KEY_FILE.unlink()

    #
    # If the current environment key came from
    # the saved file, remove it from this process too.
    #
    current = (
        os.environ.get(
            "ALPHA_GENOME_API_KEY",
            ""
        )
        .strip()
    )

    if (
        saved
        and current == saved
    ):
        os.environ.pop(
            "ALPHA_GENOME_API_KEY",
            None,
        )


def prompt_for_key():
    key = getpass.getpass(
        "Paste AlphaGenome API key "
        "(input hidden): "
    ).strip()

    return key


def manage_api_key():
    while True:
        print()
        print("=" * 70)
        print("AlphaGenome API Key")
        print("=" * 70)

        print(
            "Status:",
            api_key_status(),
        )

        print(
            "\nOfficial documentation:"
        )

        print(
            ALPHAGENOME_DOCS_URL
        )

        print()
        print(
            "1. Save / replace API key"
        )
        print(
            "2. Use API key for this session only"
        )
        print(
            "3. Remove saved API key"
        )
        print(
            "4. Back"
        )

        choice = input(
            "\nSelection: "
        ).strip()

        if choice == "1":
            key = prompt_for_key()

            if not key:
                print(
                    "No key entered. "
                    "Nothing changed."
                )
                continue

            save_api_key(
                key
            )

            print()
            print(
                "API key saved."
            )

            print(
                "Location:",
                API_KEY_FILE,
            )

            print(
                "Permissions: user read/write only (600)"
            )

        elif choice == "2":
            key = prompt_for_key()

            if not key:
                print(
                    "No key entered. "
                    "Nothing changed."
                )
                continue

            os.environ[
                "ALPHA_GENOME_API_KEY"
            ] = key

            print()
            print(
                "API key loaded for "
                "this pipeline session only."
            )

        elif choice == "3":
            if not API_KEY_FILE.exists():
                print(
                    "No saved API key exists."
                )
                continue

            confirmation = input(
                "Type REMOVE to delete "
                "the saved key: "
            ).strip()

            if confirmation != "REMOVE":
                print(
                    "Cancelled."
                )
                continue

            remove_saved_api_key()

            print(
                "Saved API key removed."
            )

        elif choice == "4":
            return

        else:
            print(
                "Invalid selection."
            )