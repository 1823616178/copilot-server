"""Offline regression for a first login without an existing profile."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from copilot.auth import load_auth


class FirstLoginTests(unittest.TestCase):
    def test_first_login_opens_browser_when_profile_does_not_exist(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("copilot.auth.can_open_browser", return_value=True),
            patch("copilot.browser.BrowserCopilot") as browser,
        ):
            token_path = str(Path(directory) / "token.json")
            profile_path = str(Path(directory) / "profile")
            expected = {"access_token": "test-token"}
            browser.return_value.login.return_value = expected

            result = load_auth(path=token_path, profile_dir=profile_path)

            self.assertEqual(result, expected)
            browser.assert_called_once_with(
                profile_dir=profile_path, headless=False, proxy=None,
            )
            browser.return_value.login.assert_called_once_with(path=token_path)
