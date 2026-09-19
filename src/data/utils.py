from typing import List, Union


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


HATEFUL_MEMES_PROMPT = "Is this image hateful?"

REPLACEMENTS_TRAIN = {
    "<|image_1|>\nRepresent the given image with the following question:": "VQA:",
    "<|image_1|>\nRepresent the given image for classification": "Classification",
    "<|image_1|>\nRepresent the given news image with the following caption for domain classification:": "Domain classification:",
    "<|image_1|>\nRepresent the given image for binary classification to determine whether it constitutes hateful speech or not": HATEFUL_MEMES_PROMPT,
    "Represent the given dialogue about an image, which is used for image retrieval:": "Image retrieval:",
    "<|image_1|>\nFind an image to match the fashion image and style note:": "Image retrieval:",
    "<|image_1|>\nRepresent the given image.": "",
    "<|image_1|>\nRepresent the given image": "",
    "<|image_1|>\nGiven an image, find a similar everyday image with the described changes:": "Change:",
    "<|image_1|>\nFind a day-to-day image that looks similar to the provided image.": "Similar image",
    "<|image_1|>\nFind a Wikipedia image that answers this question:": "Image retrieval:",
    "<|image_1|>\nRepresent the given Wikipedia image with related text information: ": "",
    "<|image_1|>\nSelect the portion of the image that isolates the object labeled as": "Object isolation:",
    "<|image_1|>\nRepresent the given cropped image of the object": "Object isolation",
    "<|image_1|>\nFind an image caption describing the given everyday image.": "Captioning",
    "Find me an everyday image that matches the given caption: ": "Image retrieval: ",
    "<|image_1|>\nIdentify the scene shown in the image": "Scene identification",
    "<|image_1|>\nIdentify the object shown in the image": "Object identification",
    "<|image_1|>\nFind a caption for the news in the given photo.": "Captioning",
    "Retrieve an image of this news caption.": "Image retrieval:",
}

REPLACEMENTS_EVAL = {
    "<|image_1|>\nIdentify the country depicted in the image": "In which country is this?",
    "<|image_1|>\nFind a news image that matches the provided caption:": "Image retrieval:",
    "<|image_1|>\nCrop the image to to isolate the object labeled as": "Object isolation:",
    "<|image_1|>\nRetrieve a Wikipedia image-description pair that provides evidence for the question of this image:": "Image retrieval:",
    "<|image_1|>\nSelect the portion of the image that follows the language expressions.": "Object isolation:",
    "<|image_1|>\nSelect the portion of the image that follows the language expressions:": "Object isolation:",
    "<|image_1|>\nSelect the portion of the image that answers the question": "Object isolation:",
    "Find the document image that can answer the given query:": "Image retrieval:",
}
REPLACEMENTS_EVAL.update(REPLACEMENTS_TRAIN)

PREFIXES_RIGHT = {
    "ImageNet-1K": "a photo of a ",
    "ImageNet_1K": "a photo of a ",
    "CIFAR-100": "a photo of a ",
    "CIFAR_100": "a photo of a ",
    "N24News": "a photo from the domain of ",
    "VOC2007": "a photo of a ",
    "SUN397": "a photo of a ",
    "Place365": "a photo of a ",
    "ImageNet-A": "a photo of a ",
    "ImageNet-R": "a photo of a ",
    "ObjectNet": "a photo of a ",
    "Country211": "this is in the country of ",
}


class TxtModifier:
    def __init__(
        self,
        model_name: str,
        is_train: bool,
        empty_prompt: Union[bool, List[str]] = False,
        drop_text: bool = False,
        prompt_prefixes: dict | None = None,
    ):
        del model_name
        self.replacements = REPLACEMENTS_TRAIN.copy() if is_train else REPLACEMENTS_EVAL.copy()
        self._set_replacements(empty_prompt)
        self.drop_text = drop_text
        self.prompt_prefixes = prompt_prefixes.copy() if prompt_prefixes is not None else None

    def _set_replacements(self, empty_prompt: Union[bool, List[str]] = False) -> None:
        replacements = {}
        if isinstance(empty_prompt, bool) and empty_prompt:
            replacements = {
                key: ""
                for key, value in self.replacements.items()
                if value != HATEFUL_MEMES_PROMPT
            }
        elif isinstance(empty_prompt, list):
            replacements = {
                key: "" for key, value in self.replacements.items() if value in empty_prompt
            }
        self.replacements.update(replacements)

    def __call__(self, txt: str, subset: str) -> str:
        if self.drop_text:
            return ""

        for old, new in self.replacements.items():
            txt = txt.replace(old, new)
        if subset in ["ImageNet-1K", "ImageNet_1K"]:
            txt = txt.split(",")[0]
        if self.prompt_prefixes is not None:
            txt = f"{self.prompt_prefixes.get(subset, '')}{txt}"
        return txt.rstrip("\n").strip()
