from abc import ABC, abstractmethod
import torch
from dataclasses import dataclass
import yaml
import csv
from tqdm import tqdm
import random
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from tokenizers import Tokenizer
from typing import *
from spacy.lang.en import English
import logging
from pathlib import Path
from functools import partial
import numpy as np
from enum import Enum
from data.dataset_utils import *

logger = logging.getLogger(__name__)


class EncodeMethod(Enum):
    SENTENCE_SPLITTING = "sentence_splitting"
    CONTEXT_LENGTH = "context_length"
    NONE = "none"


@dataclass
class PretrainingDataConfig:
    tokenizer: Tokenizer
    dataset_info: Optional[Union[DatasetInfo, List[DatasetInfo]]]
    dataset: Optional[DatasetDict]
    sentence_splitting: bool
    context_length: Optional[int]
    sentence_end_search_size: Union[int, None]
    dataset_sampling: Union[float, List[float]]
    cache_dir: Optional[str]
    dataset_dir: Optional[str]
    validation_set_size: Optional[int]
    test_set_size: Optional[int]
    training_set_size: Union[int, float]
    training_set_random_seed: int
    valid_test_split_random_seed: int
    num_proc: int
    subset_ratio: Optional[float]

    def __post_init__(self):
        if self.dataset_info is None and self.dataset is None:
            raise ValueError("Either dataset info or dataset should be provided")

        if not self.sentence_splitting and self.context_length is None:
            raise ValueError("Either sentence splitting or context length should be enabled")

        if self.context_length is not None and self.sentence_end_search_size is None:
            self.sentence_end_search_size = 0
            logger.warning("Sentence end search size is set to 0. This may lead to splitting sentences in the middle.")


class AbstractPretrainingData(ABC):
    @abstractmethod
    def make_datasets(self) -> Tuple[torch.utils.data.dataset.Dataset,
                                     torch.utils.data.dataset.Dataset,
                                     torch.utils.data.dataset.Dataset]:
        pass


    @abstractmethod
    def encode_by_batch(self,
                        documents: Dict[str, List],
                        tokenizer: Tokenizer,
                        nlp: English,
                        dataset_info: DatasetInfo,
                        bos_token: str,
                        sep_token: str) -> Dict[str, List]:
        """
        The structure of the tokenized data can be adjusted by creating new functions that encode the data in a different way.
        For example, the BaseCollator class can be modified to accept a different input format, and the encode_by_batch function
        can be modified to create the data in that format. In that case, the text is split according to a fixed token length
        instead of sentence splitting. The function should return a dictionary with the keys "input_ids" and if desired "label".
        """
        pass


class PretrainingData(AbstractPretrainingData):
    """
    A class to handle the configuration and setup of pretraining data for natural language processing models.

    Attributes:
    -----------
    tokenizer : Tokenizer
        An instance of a tokenizer used to preprocess the text data.
    dataset : Optional[DatasetDict]
        An optional dataset dictionary containing the datasets for training, validation, and testing.
    dataset_info : Optional[Union[DatasetInfo, List[DatasetInfo]]]
        Information about the dataset, which could be a single DatasetInfo object or a list of them.
    sentence_splitting : bool
        A flag indicating whether to perform sentence splitting on the text data.
    context_length : Union[bool, None]
        Defines the context length for the text data. Could be a boolean or None.
    sentence_end_search_size : Optional[int]
        The search size for finding the end of sentences during sentence splitting.
    dataset_sampling : Union[float, List[float]]
        The fraction or list of fractions representing the portion of the dataset to sample for training.
    cache_dir : str
        Directory path where cached files are stored.
    dataset_dir : str
        Directory path where the dataset files are stored.
    validation_set_size : Optional[int]
        The size of the validation set, if applicable.
    test_set_size : Optional[int]
        The size of the test set, if applicable.
    training_set_size : int
        The size of the training set.
    training_set_random_seed : int
        The random seed used for creating the training set.
    valid_test_split_random_seed : int
        The random seed used for splitting the dataset into validation and test sets.
    num_proc : int
        The number of processes to use for data preprocessing.
    subset_ratio : Optional[float]
        The ratio of the dataset to use as a subset for training, if applicable. Important: This reduces the dataset before
        the mapping step, which can be more efficient for large datasets. Only use it for debugging, because the
        subset dataset will be cached and used for all future runs.
    """

    def __init__(self, pretraining_data_config: PretrainingDataConfig):

        self.tokenizer: Tokenizer = pretraining_data_config.tokenizer
        self.dataset: Optional[DatasetDict] = pretraining_data_config.dataset
        self.dataset_info: Optional[Union[DatasetInfo, List[DatasetInfo]]] = pretraining_data_config.dataset_info
        self.sentence_splitting: bool = pretraining_data_config.sentence_splitting
        self.context_length: Union[bool, None] = pretraining_data_config.context_length#
        self.sentence_end_search_size: Optional[int] = pretraining_data_config.sentence_end_search_size
        self.dataset_sampling: Union[float, List[float]] = pretraining_data_config.dataset_sampling
        self.cache_dir: str = pretraining_data_config.cache_dir
        self.dataset_dir: str = pretraining_data_config.dataset_dir
        self.validation_set_size: Optional[int] = pretraining_data_config.validation_set_size
        self.test_set_size: Optional[int] = pretraining_data_config.test_set_size
        self.training_set_size: int = pretraining_data_config.training_set_size
        self.training_set_random_seed: int = pretraining_data_config.training_set_random_seed
        self.valid_test_split_random_seed: int = pretraining_data_config.valid_test_split_random_seed
        self.num_proc: int = pretraining_data_config.num_proc
        self.subset_ratio: Optional[float] = pretraining_data_config.subset_ratio

    def _collect_datasets(self,
                          dataset_info: Union[DatasetInfo, List[DatasetInfo]],
                          sentence_splitting: bool,
                          dataset_sampling: Union[List[float], float],
                          cache_dir: str,
                          dataset_dir: str,
                          subset_ratio: float,
                          num_proc: int) -> List[DatasetDict]:

        if isinstance(dataset_info, DatasetInfo):
            dataset_info = [dataset_info]
        PretrainingData._validate_dataset_infos(dataset_infos=dataset_info)

        if isinstance(dataset_sampling, float):
            dataset_sampling = [dataset_sampling]

        assert len(dataset_info) == len(dataset_sampling)

        logger.info(f"Dataset preprocessing started")
        combined_datasets = []
        for d_info, d_sampling in zip(dataset_info, dataset_sampling):
            encode_method: EncodeMethod = EncodeMethod.SENTENCE_SPLITTING if sentence_splitting \
                else EncodeMethod.CONTEXT_LENGTH
            cache_path: Path = PretrainingData._get_cache_path(dataset_info=d_info, cache_dir=cache_dir,
                                                               encode_method=encode_method, context_length=self.context_length)

            if cache_path.exists():
                logger.info(f"Load a cached dataset from {cache_path}")
                processed_datasets = DatasetDict.load_from_disk(dataset_dict_path=str(cache_path))
                logger.info(f"Load completed")

            else:
                # If in the dataset info the local path is provided, then load the dataset from the local path
                if d_info.save_local_path is not None:
                    logger.info(
                        f"Load unprocessed dataset from {d_info.name} using local path {d_info.save_local_path}")
                    datasets: DatasetDict = load_from_disk(d_info.save_local_path)
                    logger.info(f"Load unprocessed dataset from disk completed")

                elif dataset_dir:
                    logger.info(f"Load unprocessed dataset from {d_info.name} using cache {dataset_dir}")
                    datasets: DatasetDict = load_dataset(d_info.name, d_info.subset,
                                                         cache_dir=f"{dataset_dir}\\.cache\\huggingface\\datasets", )
                else:
                    logger.info(f"Load unprocessed dataset from {d_info.name} "
                                f"{d_info.subset if d_info.subset else ''}")
                    datasets: DatasetDict = load_dataset(d_info.name, d_info.subset)

                assert isinstance(datasets, DatasetDict)
                logger.info(f"Load unprocessed dataset completed")

                # Encode the dataset
                # WARNING this step takes 8.30 for sentence segmentation + tokenization of Wikipedia dataset
                logger.info(f"Dataset preprocessing")

                if subset_ratio is not None:
                    logger.info(f"Dataset subsampling started")
                    for d_name in datasets:
                        datasets[d_name] = datasets[d_name].select(
                            np.arange(int(len(datasets[d_name]) * subset_ratio)))
                    logger.info(f"Dataset subsampling completed")

                partial_fn: Callable = id

                if not sentence_splitting:
                    partial_fn = partial(self.encode_by_batch_context,
                                         tokenizer=self.tokenizer,
                                         max_length=self.context_length,
                                         search_area_size=self.sentence_end_search_size,
                                         text_columns=d_info.text_columns)

                else:
                    partial_fn = partial(self.encode_by_batch,
                                         tokenizer=self.tokenizer,
                                         nlp=self.get_sentence_segmentation_model(),
                                         dataset_info=d_info)

                logger.info(f"Start mapping the dataset")
                datasets: DatasetDict = datasets.map(function=partial_fn,
                                                     batched=True,
                                                     num_proc=num_proc,
                                                     remove_columns=d_info.text_columns)
                logger.info(f"Mapping completed")
                processed_datasets = DatasetDict()
                processed_datasets["train"] = datasets["train"]

                if d_info.validation_set_names:
                    processed_datasets["validation"] = concatenate_datasets([datasets[name]
                                                                             for name in d_info.validation_set_names])

                if d_info.test_set_names:
                    processed_datasets["test"] = concatenate_datasets([datasets[name]
                                                                       for name in d_info.test_set_names])

                # Remove all features which are not required by the models
                # to allow the concatenation across different datasets
                nested_features = [v for _, v in processed_datasets.column_names.items()]
                flatten_features = [item for items in nested_features for item in items]
                extra_cols = set(flatten_features) - {"input_ids", "label"}
                processed_datasets.remove_columns(list(extra_cols))

                logger.info(f"Cache this processed dataset into {cache_path}")
                processed_datasets.save_to_disk(dataset_dict_path=str(cache_path))
                # Workaround to force the processed_dataset to remove the extra columns
                processed_datasets = DatasetDict.load_from_disk(dataset_dict_path=str(cache_path))
                logger.info(f"Cache completed")

            if d_sampling < 1.0:
                logger.info(f"Dataset downsampling started")
                for d_name in processed_datasets:
                    processed_datasets[d_name] = processed_datasets[d_name].select(
                        np.arange(int(len(processed_datasets[d_name]) * d_sampling)))
                logger.info(f"Dataset downsampling completed")

            combined_datasets += [processed_datasets]

        logger.info(f"Dataset preprocessing completed")
        return combined_datasets

    def make_datasets(self) -> Tuple[torch.utils.data.dataset.Dataset,
                                     torch.utils.data.dataset.Dataset,
                                     torch.utils.data.dataset.Dataset]:
        """
        This function is used to create the train, validation and test datasets. It does all the necessary preprocessing
        to create the datasets. It also allows to create a subset of the dataset for faster training.

        :param subset_ratio:
        :param tokenizer: Already trained tokenizer
        :param dataset_info: DatasetInfo or list of DatasetInfo to be used to create the train, val and test sets
        :param dataset_sampling:
        :param dataset_dir: Directory path for the cache for the HuggingFace datasets library
        :param cache_dir: Directory to store the cache for the processed datasets.
        :param validation_set_size: Validation set size.
        :param test_set_size: Test set size. If None, then use the dataset["test"]. Default None.
        :param training_set_size: Default -1 to use all possible samples
        :param training_set_random_seed: Seed used only for shuffle the training set
        :param valid_test_split_random_seed: Seed used only for the split between test and validation sets.
            Required to ensure the validation set remains the same if seed is used.
        :param num_proc: Number of processor for the creation of the dataset
        :return: train dataset, validation dataset, test dataset
        """

        combined_datasets: List[DatasetDict] = self._collect_datasets(dataset_info=dataset_info,
                                                                      dataset_sampling=self.dataset_sampling,
                                                                      sentence_splitting=self.sentence_splitting,
                                                                      cache_dir=self.cache_dir,
                                                                      dataset_dir=self.dataset_dir,
                                                                      subset_ratio=self.subset_ratio,
                                                                      num_proc=self.num_proc)

        train_set = [combined_dataset["train"] for combined_dataset in combined_datasets]
        if len(train_set) > 1:
            train_set = concatenate_datasets(train_set)
        else:
            train_set = train_set[0]

        val_combined_dataset = [combined_dataset["validation"] for combined_dataset in combined_datasets
                                if "validation" in combined_dataset]
        if len(val_combined_dataset) > 1:
            val_set = concatenate_datasets(val_combined_dataset)
        elif len(val_combined_dataset) == 1:
            val_set = val_combined_dataset[0]
        else:
            val_set = None

        test_combined_dataset = [combined_dataset["test"] for combined_dataset in combined_datasets
                                 if "test" in combined_dataset]
        if len(test_combined_dataset) > 1:
            test_set = concatenate_datasets(test_combined_dataset)
        elif len(test_combined_dataset) == 1:
            test_set = test_combined_dataset[0]
        else:
            test_set = None

        assert len(train_set) > 0, "Your train set is empty"

        if self.test_set_size is None and self.validation_set_size is not None:
            # Case you extract a validation set from train set and no test set
            subset_indices = [self.validation_set_size, len(train_set) - self.validation_set_size]

            generator = torch.Generator().manual_seed(self.valid_test_split_random_seed)
            val_set, train_set = torch.utils.data.random_split(dataset=train_set,
                                                               lengths=subset_indices,
                                                               generator=generator)
        elif self.test_set_size is not None and self.validation_set_size is not None:
            # Case you extract a test and validation set from train set
            subset_indices = [self.test_set_size,
                              self.validation_set_size]

            if len(train_set) - sum(subset_indices) < 0:
                raise ValueError("The test and validation set size is larger than the training set size. "
                                 "Your subset ratios have failed and do not align with the test and train size")

            subset_indices += [len(train_set) - sum(subset_indices)]

            generator = torch.Generator().manual_seed(self.valid_test_split_random_seed)
            test_set, val_set, train_set = torch.utils.data.random_split(dataset=train_set,
                                                                         lengths=subset_indices,
                                                                         generator=generator)
        else:
            # Case you don't need to use data from train set for validation and/or test sets
            train_set = train_set.shuffle(seed=self.training_set_random_seed)


        assert val_set is not None and len(val_set) > 0, "Your validation set is empty"

        logger.info(f"Training size: {len(train_set)}")
        logger.info(f"Valid size: {len(val_set)}")

        if test_set:
            logger.info(f"Test size: {len(test_set)}")

        return train_set, val_set, test_set

    @staticmethod
    def _get_sentence_segmentation_model():
        """
        Util function to create the sentence segmentation model
        :return:
        """
        # Load Spacy model
        nlp = English()
        # nlp.max_length = 1000000 * 8 # Equivalent of 8 GB of memory / Required to have all text in memory
        # sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe("sentencizer")
        return nlp

    @staticmethod
    def search_for_sentence_end(words: List[str], end_index: int, max_length: int) -> int:
        """
        Search for the end of the sentence in the text
        :param text: Text to search for the end of the sentence
        :return: List of indexes where the sentence ends
        """
        sentence_end_punctuations = {".", "!", "?"}

        add_on = 0
        while end_index < max_length:
            if end_index >= len(words):
                return add_on
            if any(x in sentence_end_punctuations for x in words[end_index]):
                # Check if it's a typical sentence end or a special case like an abbreviation/number
                if end_index < len(words) - 1:
                    if not (words[end_index] == "." and words[end_index - 1].isnumeric()):
                        return add_on
                else:
                    # If it's the last word, and it's a punctuation, it's the end of a sentence
                    return end_index
            end_index += 1
            add_on += 1

        return add_on

    #TODO: It is tricky to set max context length in terms of words when they are converted
    @staticmethod
    def encode_by_batch_context(dataset: Dict[str, List],
                                 tokenizer: Tokenizer,
                                 max_length: int,
                                 search_area_size: int,
                                 text_columns: Union[List[str], str],
                                 bos_token="<BOS>",
                                 sep_token="<SEP>",
                                 ) -> Dict[str, List]:
        """
        This function is used to encode the documents by batch. It will split the documents into chunks of max_length
        and tokenize them. If desired, it will also search for the end of the sentence to avoid splitting the sentence
        :param dataset: Dataset to be encoded
        :param tokenizer: Tokenizer to be used
        :param max_length: Maximum length of the tokenized document
        :param search_area_size:  Size of the search area to find the end of the sentence. If 0, then no search is done
        :param dataset_info:
        :param bos_token:
        :param sep_token:
        :return:
        """
        encoded_docs = []

        if isinstance(text_columns, list):
            text_columns = text_columns[0]

        assert dataset[text_columns] is not None and len(dataset[text_columns]) > 0
        # Iterate over each document in the specified text column
        for i_d in range(len(dataset[text_columns])):
            text = dataset[text_columns][i_d]
            words = text.split()
            chunks = []

            if search_area_size == 0:
                for i in range(0, len(words), max_length):
                    current_words = words[i:i + max_length]
                    chunks.append(' '.join(current_words))

            else:
                end_index = max_length - search_area_size
                while True:

                    current_words = words[:end_index]
                    sentence_end_index = PretrainingData.search_for_sentence_end(words, end_index,
                                                                                 max_length - 1)
                    current_words.extend(words[end_index:end_index + sentence_end_index])
                    current_words.insert(0, bos_token)
                    chunks.append(' '.join(current_words))
                    words = words[end_index + sentence_end_index:]

                    if len(words) < max_length:
                        chunks.append(' '.join(words))
                        break

                    end_index = max_length - search_area_size

            encoded_doc = [encoding.ids for encoding in tokenizer.encode_batch(chunks)]
            encoded_docs += [encoded_doc]

        if "label" in dataset:
            return {"input_ids": encoded_docs,
                    "label": dataset["label"]}
        else:
            return {"input_ids": encoded_docs}

    @staticmethod
    def encode_by_batch(documents: Dict[str, List],
                         tokenizer: Tokenizer,
                         nlp: English,
                         dataset_info: DatasetInfo,
                         bos_token="<BOS>",
                         sep_token="<SEP>",
                         ) -> Dict[str, List]:
        """
        Perform sentence segmentaton and tokenization.

        :param documents: List of all documents (string)
        :param tokenizer: Tokenizer
        :param bos_token: BOS token to be added
        :param sep_token: SEP token to be added
        :return:
        """
        encoded_docs = []

        assert len(dataset_info.text_columns) >= 1
        # get length of dataset of the text column
        # insert the correct column in dataset info object
        for i_d in range(len(documents[dataset_info.text_columns[0]])):
            if dataset_info.sentence_segmentation:
                assert len(dataset_info.text_columns) == 1, dataset_info
                # sents creates a generator
                d = [s.text for s in nlp(documents[dataset_info.text_columns[0]][i_d]).sents]
            else:
                d = [documents[text_column][i_d] for text_column in dataset_info.text_columns]

            texts = [s + sep_token for s in d]
            # add CLS token to the beginning
            texts[0] = bos_token + texts[0]
            # the encoded doc contains of n senteces, the first sentence has a BOS/CLS token at the beginning
            encoded_doc = [encoding.ids for encoding in tokenizer.encode_batch(texts)]
            encoded_docs += [encoded_doc]

        if "label" in documents:
            return {"input_ids": encoded_docs,
                    "label": documents["label"]}
        else:
            return {"input_ids": encoded_docs}

    @staticmethod
    def _validate_dataset_infos(dataset_infos: List[DatasetInfo]):
        """
            Validate if the list is correct
        :param dataset_infos:
        :return:
        """

        assert isinstance(dataset_infos, List)
        assert len(dataset_infos) >= 1

        assert all([dataset_info.is_pretraining for dataset_info in dataset_infos]) or \
               (len(dataset_infos) == 1 and dataset_infos[0].is_downstream)

    @staticmethod
    def _get_cache_path(dataset_info: DatasetInfo,
                        cache_dir: str,
                        encode_method: EncodeMethod,
                        context_length: Optional[int]) -> Path:
        """
        :param dataset_info:
        :param cache_dir:
        :return: the path for the cache file for the processed dataset
        """
        cache_path: Path = Path(cache_dir if cache_dir else "")
        cache_path /= "data"
        cache_path /= str(encode_method.value) + (
            str(context_length) if encode_method == EncodeMethod.CONTEXT_LENGTH else "")
        cache_path /= f"{dataset_info.name}-{dataset_info.subset}.cache"

        return cache_path

    @staticmethod
    def get_sentence_segmentation_model():
        """
        Util function to create the sentence segmentation model
        :return:
        """
        # Load Spacy model
        nlp = English()
        # nlp.max_length = 1000000 * 8 # Equivalent of 8 GB of memory / Required to have all text in memory
        # sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe("sentencizer")
        return nlp


if __name__ == "__main__":
    from tokenizer.tokenizer_training import TokenizerUtility
    from collator import BaseCollator, CollatorConfig
    from torch.utils.data import DataLoader
    tokenizer = TokenizerUtility.get_tokenizer("../tokenizer/trained_tokenizer/ByteLevelBPETokenizer-vocab_size=30522-min_frequency=2")
    dataset_info = DatasetUtils.dataset_info_from_yaml("dataset_infos.yaml")
    pretraining_data_config = PretrainingDataConfig(tokenizer=tokenizer,
                                                    dataset_info=dataset_info,
                                                    dataset=None,
                                                    sentence_splitting=False,
                                                    context_length=300,
                                                    sentence_end_search_size=10,
                                                    dataset_sampling=1.0,
                                                    cache_dir=None,
                                                    dataset_dir=None,
                                                    validation_set_size=1000,
                                                    test_set_size=1000,
                                                    training_set_size=4000,
                                                    training_set_random_seed=42,
                                                    valid_test_split_random_seed=42,
                                                    num_proc=1,
                                                    subset_ratio=0.001) # only for debugging

    pretraining_data = PretrainingData(pretraining_data_config)
    ds = pretraining_data.make_datasets()
    collator_config = CollatorConfig(tokenizer=tokenizer,
                                     is_downstream=False,
                                     pad_token_id=tokenizer.token_to_id("<PAD>"),
                                     text_column="input_ids",
                                     )

    collator = BaseCollator(collator_config)
    train_set = ds[0]
    #TODO: outputs wrong batch size with data laoder
    train_loader = DataLoader(train_set, batch_size=8, collate_fn=collator, num_workers=2)
    collator([train_set[0], train_set[1]])