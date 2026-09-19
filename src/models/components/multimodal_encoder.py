"""Text encoder construction for the shared X-JEPA encoder envelope."""

import logging

import transformers
from transformers import BertLMHeadModel, BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig

transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)


def init_tokenizer(size: str, truncation_side: str = "right") -> BertTokenizer:
    tokenizer = BertTokenizer.from_pretrained(size, truncation_side=truncation_side)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    return tokenizer


def init_text_encoder(size: str, load_pretrained: bool = True) -> BertLMHeadModel:
    """Return a BERT-base text encoder, optionally initialized from pretrained weights."""
    encoder_config = BertConfig.from_pretrained(size)

    if load_pretrained:
        logger.info("Loading pretrained BERT encoder of size `%s`.", size)
        return BertLMHeadModel.from_pretrained(size, config=encoder_config)

    logger.info("Initializing BERT encoder of size `%s` with random weights.", size)
    return BertLMHeadModel(config=encoder_config)
