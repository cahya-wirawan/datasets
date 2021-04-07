# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
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
""" OpenSLR Dataset"""

from __future__ import absolute_import, division, print_function

import os
import re
from pathlib import Path

import datasets


_DATA_URL = "https://openslr.org/resources/{}"

_CITATION = """\

"""

_DESCRIPTION = """\
OpenSLR is a site devoted to hosting speech and language resources, such as training corpora for speech recognition,
and software related to speech recognition. We intend to be a convenient place for anyone to put resources that
they have created, so that they can be downloaded publicly.
"""

_HOMEPAGE = "https://openslr.org/"

_LICENSE = ""

_RESOURCES = {
    "SLR35": {
        "Language": "Javanese",
        "LongName": "Large Javanese ASR training data set",
        "Category": "Speech",
        "Summary": "Javanese ASR training data set containing ~185K utterances",
        "Files": ["asr_javanese_0.zip", "asr_javanese_1.zip", "asr_javanese_2.zip", "asr_javanese_3.zip",
                  "asr_javanese_4.zip", "asr_javanese_5.zip", "asr_javanese_6.zip", "asr_javanese_7.zip",
                  "asr_javanese_8.zip", "asr_javanese_9.zip", "asr_javanese_a.zip", "asr_javanese_b.zip",
                  "asr_javanese_c.zip", "asr_javanese_d.zip", "asr_javanese_e.zip", "asr_javanese_f.zip"],
        "IndexFiles": ["asr_javanese/utt_spk_text.tsv"]*16,
        "DataDirs": ["asr_javanese/data"]*16,
    },
    "SLR36": {
        "Language": "Sundanese",
        "LongName": "Large Sundanese ASR training data set",
        "Category": "Speech",
        "Summary": "Sundanese ASR training data set containing ~220K utterances",
        "Files": ["asr_sundanese_0.zip", "asr_sundanese_1.zip", "asr_sundanese_2.zip", "asr_sundanese_3.zip",
                  "asr_sundanese_4.zip", "asr_sundanese_5.zip", "asr_sundanese_6.zip", "asr_sundanese_7.zip",
                  "asr_sundanese_8.zip", "asr_sundanese_9.zip", "asr_sundanese_a.zip", "asr_sundanese_b.zip",
                  "asr_sundanese_c.zip", "asr_sundanese_d.zip", "asr_sundanese_e.zip", "asr_sundanese_f.zip"],
        "IndexFiles": ["asr_sundanese/utt_spk_text.tsv"]*16,
        "DataDirs": ["asr_sundanese/data"]*16,
    },
    "SLR41": {
        "Language": "Javanese",
        "LongName": "High quality TTS data for Javanese",
        "Category": "Speech",
        "Summary": "Multi-speaker TTS data for Javanese (jv-ID)",
        "Files": ["jv_id_female.zip", "jv_id_male.zip"],
        "IndexFiles": ["jv_id_female/line_index.tsv", "jv_id_male/line_index.tsv"],
        "DataDirs": ["jv_id_female/wavs", "jv_id_male/wavs"],
    },
    "SLR42": {
        "Language": "Khmer",
        "LongName": "High quality TTS data for Khmer",
        "Category": "Speech",
        "Summary": "Multi-speaker TTS data for Khmer (km-KH)",
        "Files": ["km_kh_male.zip"],
        "IndexFiles": ["km_kh_male/line_index.tsv"],
        "DataDirs": ["km_kh_male/wavs"],
    },
    "SLR43": {
        "Language": "Nepali",
        "LongName": "High quality TTS data for Nepali",
        "Category": "Speech",
        "Summary": "Multi-speaker TTS data for Nepali (ne-NP)",
        "Files": ["ne_np_female.zip"],
        "IndexFiles": ["ne_np_female/line_index.tsv"],
        "DataDirs": ["ne_np_female/wavs"],
    },
    "SLR44": {
        "Language": "Sundanese",
        "LongName": "High quality TTS data for Sundanese",
        "Category": "Speech",
        "Summary": "Multi-speaker TTS data for Javanese Sundanese (su-ID)",
        "Files": ["su_id_female.zip", "su_id_male.zip"],
        "IndexFiles": ["su_id_female/line_index.tsv", "su_id_male/line_index.tsv"],
        "DataDirs": ["su_id_female/wavs", "su_id_male/wavs"],
    },
    "SLR63": {
        "Language": "Malayalam",
        "LongName": "Crowdsourced high-quality Malayalam multi-speaker speech data set",
        "Category": "Speech",
        "Summary": "Data set which contains recordings of native speakers of Malayalam",
        "Files": ["ml_in_female.zip", "ml_in_male.zip"],
        "IndexFiles": ["line_index.tsv", "line_index.tsv"],
        "DataDirs": ["", ""],
    },
    "SLR64": {
        "Language": "Marathi",
        "LongName": "Crowdsourced high-quality Marathi multi-speaker speech data set",
        "Category": "Speech",
        "Summary": "Data set which contains recordings of native speakers of Marathi",
        "Files": ["mr_in_female.zip"],
        "IndexFiles": ["line_index.tsv"],
        "DataDirs": ["", ""],
    },
    "SLR65": {
        "Language": "Tamil",
        "LongName": "Crowdsourced high-quality Tamil multi-speaker speech data set",
        "Category": "Speech",
        "Summary": "Data set which contains recordings of native speakers of Tamil",
        "Files": ["ta_in_female.zip", "ta_in_male.zip"],
        "IndexFiles": ["line_index.tsv", "line_index.tsv"],
        "DataDirs": ["", ""],
    },
    "SLR66": {
        "Language": "Telugu",
        "LongName": "Crowdsourced high-quality Telugu multi-speaker speech data set",
        "Category": "Speech",
        "Summary": "Data set which contains recordings of native speakers of Telugu",
        "Files": ["te_in_female.zip", "te_in_male.zip"],
        "IndexFiles": ["line_index.tsv", "line_index.tsv"],
        "DataDirs": ["", ""],
    },
    "SLR69": {
        "Language": "Catalan",
        "LongName": "Crowdsourced high-quality Catalan speech data set",
        "Category": "Speech",
        "Summary": "Data set which contains recordings of Catalan",
        "Files": ["ca_es_female.zip", "ca_es_male.zip"],
        "IndexFiles": ["line_index.tsv", "line_index.tsv"],
        "DataDirs": ["", ""],
    },
}


class OpenSlrConfig(datasets.BuilderConfig):
    """BuilderConfig for OpenSlr."""

    def __init__(self, name, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        self.language = kwargs.pop("language", None)
        self.long_name = kwargs.pop("long_name", None)
        self.category = kwargs.pop("category", None)
        self.summary = kwargs.pop("summary", None)
        self.files = kwargs.pop("files", None)
        self.index_files = kwargs.pop("index_files", None)
        self.data_dirs = kwargs.pop("data_dirs", None)
        description = f"Open Speech and Language Resources dataset in {self.language}. Name: {self.name}, Summary: {self.summary}."
        super(OpenSlrConfig, self).__init__(name=name, description=description, **kwargs)


class OpenSlr(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        OpenSlrConfig(
            name=resource_id,
            language=_RESOURCES[resource_id]["Language"],
            long_name=_RESOURCES[resource_id]["LongName"],
            category=_RESOURCES[resource_id]["Category"],
            summary=_RESOURCES[resource_id]["Summary"],
            files=_RESOURCES[resource_id]["Files"],
            index_files=_RESOURCES[resource_id]["IndexFiles"],
            data_dirs=_RESOURCES[resource_id]["DataDirs"],
        )
        for resource_id in _RESOURCES.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "path": datasets.Value("string"),
                "sentence": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        resource_number = self.config.name.replace("SLR", "")
        urls = [f"{_DATA_URL.format(resource_number)}/{file}" for file in self.config.files]
        dl_paths = dl_manager.download_and_extract(urls)
        abs_path_to_indexs = [os.path.join(path, f"{self.config.index_files[i]}") for i, path in enumerate(dl_paths)]
        abs_path_to_datas = [os.path.join(path, f"{self.config.data_dirs[i]}") for i, path in enumerate(dl_paths)]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path_to_indexs": abs_path_to_indexs,
                    "path_to_datas": abs_path_to_datas,
                },
            ),
        ]

    def _generate_examples(self, path_to_indexs, path_to_datas):
        """ Yields examples. """

        counter = -1
        if self.config.name in ["SLR35", "SLR36"]:
            sentence_index = {}
            for i, path in enumerate(path_to_indexs):
                with open(path, encoding="utf-8") as f:
                    lines = f.readlines()
                    for id_, line in enumerate(lines):
                        field_values = re.split(r"\t\t?", line.strip())
                        filename, user_id, sentence = field_values
                        sentence_index[filename] = sentence
                for p in Path(path_to_datas[i]).rglob("*.flac"):
                    filename = p.stem
                    if p.stem not in sentence_index:
                        continue
                    path = str(p.resolve())
                    sentence = sentence_index[filename]
                    counter += 1
                    yield counter, {"path": path, "sentence": sentence}
        else:
            for i, path in enumerate(path_to_indexs):
                with open(path, encoding="utf-8") as f:
                    lines = f.readlines()
                    for id_, line in enumerate(lines):
                        # Following regexs are needed to normalise the lines, since the datasets
                        # are not always consistent and have bugs:
                        line = re.sub(r"\t[^\t]*\t", "\t", line.strip())
                        field_values = re.split(r"\t\t?", line)
                        if len(field_values) != 2:
                            continue
                        filename, sentence = field_values
                        # set absolute path for audio file
                        path = os.path.join(path_to_datas[i], f"{filename}.wav")
                        counter += 1
                        yield counter, {"path": path, "sentence": sentence}
