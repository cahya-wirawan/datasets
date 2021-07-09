# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Large-scale Indonesian Summarization Dataset"""


import glob
import json
import os
import re
from pathlib import Path

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\

"""

_DESCRIPTION = """\
This module load text dataset from local directory. The text dataset should have the format like Oscar dataset
where each new entry is separated by empty lines.
"""

_HOMEPAGE = ""

_LICENSE = ""


class IdCollectionConfig(datasets.BuilderConfig):
    """BuilderConfig for IdCollection"""

    def __init__(self, **kwargs):
        """BuilderConfig for IdCollection.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(IdCollectionConfig, self).__init__(**kwargs)


class IdCollection(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        IdCollectionConfig(
            name="id_collection",
            version=VERSION,
            description="Id Collection dataset",
        ),
    ]

    @property
    def manual_download_instructions(self):
        return """\
            You need to manually collect text datasets in a directory.  The text dataset can then be loaded 
            using the following command:
            `datasets.load_dataset("id_collection", data_dir="<path/to/dataset>")`.
            """

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"id": datasets.Value("int64"), "text": datasets.Value("string")}),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        print("# Data directory", data_dir)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                "{} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('id_liputan6', "
                "'canonical', data_dir=...)`. Manual download instructions:\n{}".format(
                    data_dir, self.manual_download_instructions
                )
            )
        split_generators = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "article_dir": os.path.join(data_dir, ""),
                    "split": "train",
                },
            )
        ]
        return split_generators

    def _generate_examples(self, article_dir, split):
        logger.info("⏳ Generating %s examples from = %s", split, article_dir)
        id_ = 0
        current_lines = []
        for path in sorted(glob.glob(os.path.join(article_dir, "**/*.txt"), recursive=True)):
            with open(path, "r") as f:
                print("# Reading", path)
                for line in f:
                    if len(line.strip()) > 0:
                        current_lines.append(line)
                    elif current_lines:
                        feature = id_, {"id": id_, "text": "".join(current_lines).rstrip()}
                        yield feature
                        id_ += 1
                        current_lines = []
                # last paragraph
                if current_lines:
                    feature = id_, {"id": id_, "text": "".join(current_lines).rstrip()}
                    yield feature
                    id_ += 1
                    current_lines = []
