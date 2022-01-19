# coding=utf-8
import json
import os
import csv
import re

import datasets

from PIL import Image
import numpy as np

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{yu2021pick,
               title={PICK: Processing key information extraction from documents using improved graph learning-convolutional networks},
               author={Yu, Wenwen and Lu, Ning and Qi, Xianbiao and Gong, Ping and Xiao, Rong},
               booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
               pages={4363--4370},
               year={2021},
               organization={IEEE}
}
"""
_DESCRIPTION = """\
The train ticket is fixed layout dataset, however, it contains background noise and imaging distortions.
It contains 1,530 synthetic images and 320 real images for training, and 80 real images for testing.
Every train ticket has eight key text fields including ticket number, starting station, train number, destination station, date, ticket rates, seat category, and name.
This dataset mainly consists of digits, English characters, and Chinese characters.
"""

_URL = """\
https://drive.google.com/file/d/1o8JktPD7bS74tfjz-8dVcZq_uFS6YEGh/view?usp=sharing
"""


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


class TrainTicketsConfig(datasets.BuilderConfig):
    """BuilderConfig for train_tickets"""

    def __init__(self, **kwargs):
        """BuilderConfig for train_tickets.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TrainTicketsConfig, self).__init__(**kwargs)


class TrainTickets(datasets.GeneratorBasedBuilder):
    """train tickets"""

    BUILDER_CONFIGS = [
        TrainTicketsConfig(
            name="train_tickets-yu2020pick",
            version=datasets.Version("1.0.0"),
            description="Chinese train tickets",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int64"))
                    ),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "S-DATE",
                                "S-DESTINATION_STATION",
                                "S-NAME",
                                "S-SEAT_CATEGORY",
                                "S-STARTING_STATION",
                                "S-TICKET_NUM",
                                "S-TICKET_RATES",
                                "S-TRAIN_NUM",
                            ]
                        )
                    ),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/wenwenyu/PICK-pytorch",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(
            "https://drive.google.com/uc?export=download&id=1o8JktPD7bS74tfjz-8dVcZq_uFS6YEGh"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filelist": f"{downloaded_file}/train_tickets/synth1530_real320_baseline_trainset.csv"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filelist": f"{downloaded_file}/train_tickets/real80_baseline_testset.csv"
                },
            ),
        ]

    # based on https://github.com/wenwenyu/PICK-pytorch/blob/master/data_utils/documents.py#L229
    def _read_gt_file_with_box_entity_type(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            document_text = f.read()

        # match pattern in document: index,x1,y1,x2,y2,x3,y3,x4,y4,transcript,box_entity_type
        regex = (
            r"^\s*(-?\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,"
            r"\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,(.*),(.*)\n?$"
        )

        matches = re.finditer(regex, document_text, re.MULTILINE)

        res = []
        for _, match in enumerate(matches, start=1):
            points = [int(match.group(i)) for i in range(2, 10)]
            x = points[0:8:2]
            y = points[1:8:2]
            x1 = min(x)
            y1 = min(y)
            x2 = max(x)
            y2 = max(y)
            transcription = str(match.group(10))
            entity_type = str(match.group(11))
            res.append((x1, y1, x2, y2, transcription, entity_type))
        return res

    def _generate_examples(self, filelist):
        logger.info("⏳ Generating examples from = %s", filelist)

        ann_dir = os.path.join(os.path.dirname(filelist), "boxes_trans")
        img_dir = os.path.join(os.path.dirname(filelist), "images1930")
        print(ann_dir)

        with open(filelist) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                guid = row[0]
                # document_type = row[1]
                filename = row[2]

                words = []
                bboxes = []
                ner_tags = []
                file_path = os.path.join(ann_dir, f"{filename}.tsv")
                data = self._read_gt_file_with_box_entity_type(file_path)
                image_path = os.path.join(img_dir, f"{filename}.jpg")
                _, size = load_image(image_path)
                for item in data:
                    box = item[0:4]
                    transcription, label = item[4:6]
                    words.append(transcription)
                    bboxes.append(normalize_bbox(box, size))
                    if label == "other":
                        ner_tags.append("O")
                    else:
                        ner_tags.append("S-" + label.upper())
                yield guid, {
                    "id": str(guid),
                    "words": words,
                    "bboxes": bboxes,
                    "ner_tags": ner_tags,
                    "image_path": image_path,
                }
