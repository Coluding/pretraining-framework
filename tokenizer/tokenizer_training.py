"""Entry point to train the tokenizer."""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from tqdm import tqdm
from typing import Iterable

import numpy as np
from datasets import load_from_disk, load_dataset

from tokenizers.implementations import BaseTokenizer, ByteLevelBPETokenizer
from tokenizers import Tokenizer


logger = logging.getLogger(__name__)


class TokenizerUtility:
    ALGORITHM_MAPPINGS = {"ByteLevelBPETokenizer": ByteLevelBPETokenizer}

    @staticmethod
    def get_unique_id(algorithm: str,
                      vocab_size: int,
                      min_frequency: int) -> str:
        """
        :param algorithm: Algorithm used to train the tokenizer
        :param vocab_size: Size of the vocabulary
        :param min_frequency: Minimum frequency of the tokens
        :return: Unique id for the tokenizer
        """
        return f"{algorithm}-vocab_size={vocab_size}-min_frequency={min_frequency}"

    @staticmethod
    def train_tokenizer(
            output_dir: str,
            dataset: str  = "wikipedia",
            algorithm: str = "ByteLevelBPETokenizer",
            vocab_size: int = 30522,
            min_frequency: int = 2,
            seed: int = 42,
            max_documents: int = 100000,
            text_column: str = "text",
            dataset_config: str = None

    ) -> None:
        """
        Main function to train tokenizers.
        Special tokens are hard-coded as following:
            "<PAD>",  # Padding values must be 0
            "<MASK>",  # Masked tokens must be 1
            "<BOS>",  # BOS must be 2
            "<EOS>",  # EOS must be 3
            "<SEP>",  # SEP must be 4
            "<UNK>",  # UNK must be 5 (not relevant for BBPE but still present in vocab)


        :param output_dir: Path where the tokenizer will be saved
        :param dataset: Name or path for the HuggingFace nlp library
        :param algorithm: ByteLevelBPETokenizer (Electra use WordPiece as BERT)
        :param vocab_size: Default 30522 like Electra
        :param min_frequency: Default 2
        :param seed: Default 42
        :param max_documents: If number of documents is higher, then subsampling to prevent OOM
        :return:
        """
        if algorithm not in TokenizerUtility.ALGORITHM_MAPPINGS:
            raise NotImplementedError(f"Algorithm {algorithm} not yet covered")

        np.random.seed(seed)
        logger.info(f"Using seed {seed}")

        tokenizer: BaseTokenizer = TokenizerUtility.ALGORITHM_MAPPINGS[algorithm]()

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        if Path(dataset).exists():
            logger.info(f"Dataset {dataset} will be loaded from disk")
            dataset = load_from_disk(dataset_path=dataset)
        else:
            logger.info(f"Dataset {dataset} is a standard HuggingFace dataset")
            dataset = load_dataset(dataset, dataset_config, split="train")
        logger.info(f"Dataset {dataset} loaded")


        if hasattr(dataset, "keys") and "train" in dataset.keys():
            dataset = dataset["train"]

        max_documents = min(max_documents, len(dataset))
        logger.info(f"max_documents {max_documents}")

        # Using all files will create OOM errors
        # Better to subsample to have roughly 3GB of data (recommendation from HF in their github is 1GB)
        f = NamedTemporaryFile(mode='w+', delete=False)
        logger.info(f"Write all documents in tmp file {f.name}")
        iterator: Iterable = tqdm(np.random.randint(low=0,
                                          high=len(dataset),
                                          size=max_documents),
                        desc="Writing documents to tmp file")

        for i in iterator:
            txt: str = dataset[int(i)][text_column]
            f.writelines([txt + "\n"])

        logger.info(f"Train tokenizer from tmp file")
        # noinspection PyUnresolvedReferences
        tokenizer.train(files=[f.name], vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[
            "<PAD>",  # Padding values must be 0
            "<MASK>",  # Masked tokens must be 1
            "<BOS>",  # BOS must be 2
            "<EOS>",  # EOS must be 3
            "<SEP>",  # SEP must be 4
            "<UNK>",  # UNK must be 5 (not relevant for BBPE but still present in vocab)
        ])

        tokenizer_unique_id: str = TokenizerUtility.get_unique_id(algorithm=algorithm,
                                                                  vocab_size=vocab_size,
                                                                  min_frequency=min_frequency)
        logger.info(tokenizer_unique_id)
        tokenizer.save(path=str(output_dir / tokenizer_unique_id), pretty=True)
        logger.info(f"Model saved in {output_dir}/{tokenizer_unique_id}")

    @staticmethod
    def get_tokenizer(tokenizer_path: str) -> Tokenizer:
        """
        Load the tokenizer from the given path
        :param tokenizer_path: Path to the tokenizer
        :return: Tokenizer
        """
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return tokenizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TokenizerUtility.train_tokenizer(output_dir="./trained_tokenizer",
                                     dataset="Skylion007/openwebtext",
                                     algorithm="ByteLevelBPETokenizer",
                                     vocab_size=30522,
                                     min_frequency=2,
                                     seed=42,
                                     max_documents=100000,
                                     text_column="text",
                                     dataset_config=None
                                     )