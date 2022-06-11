#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import unittest
from unittest import mock

from dot import DOT


def fake_generate(self, option, source, target, show_fps=False, **kwargs):
    return [[None], [None]]


@mock.patch.object(DOT, "generate", fake_generate)
class TestDotOptions(unittest.TestCase):
    def setUp(self):
        self._dot = DOT(use_image=True, save_folder="./tests")

        self.faceswap_cv2_option = self._dot.faceswap_cv2(False, False, None)

        self.fomm_option = self._dot.fomm(False, False, None)

        self.simswap_option = self._dot.simswap(False, False, None)

    def test_option_creation(self):

        success, rejected = self._dot.generate(
            self.faceswap_cv2_option,
            "./tests",
            "./tests",
            show_fps=False,
            model_path=None,
            limit=5,
        )
        assert len(success) == 1
        assert len(rejected) == 1

        success, rejected = self._dot.generate(
            self.fomm_option,
            "./tests",
            "./tests",
            show_fps=False,
            model_path=None,
            limit=5,
        )
        assert len(success) == 1
        assert len(rejected) == 1

        success, rejected = self._dot.generate(
            self.simswap_option,
            "./tests",
            "./tests",
            show_fps=False,
            parsing_model_path=None,
            arcface_model_path=None,
            checkpoints_dir=None,
            limit=5,
        )
        assert len(success) == 1
        assert len(rejected) == 1
