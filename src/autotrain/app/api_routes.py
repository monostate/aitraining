import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, get_type_hints

import torch
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi, constants
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status
from PIL import Image
from pydantic import BaseModel, ConfigDict, create_model, model_validator
from transformers import AutoModelForVision2Seq, AutoProcessor, pipeline

from autotrain import __version__, logger
from autotrain.app.inference_utils import detect_model_type, get_model_metadata
from autotrain.app.params import HIDDEN_PARAMS, PARAMS, AppParams
from autotrain.app.utils import token_verification
from autotrain.project import AutoTrainProject
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.vlm.params import VLMTrainingParams


FIELDS_TO_EXCLUDE = HIDDEN_PARAMS + ["push_to_hub"]
MODEL_CACHE = {}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


def create_api_base_model(base_class, class_name):
    annotations = get_type_hints(base_class)
    if class_name in ("LLMSFTTrainingParamsAPI", "LLMRewardTrainingParamsAPI"):
        more_hidden_params = [
            "model_ref",
            "dpo_beta",
            "add_eos_token",
            "max_prompt_length",
            "max_completion_length",
        ]
    elif class_name == "LLMORPOTrainingParamsAPI":
        more_hidden_params = [
            "model_ref",
            "dpo_beta",
            "add_eos_token",
        ]
    elif class_name == "LLMDPOTrainingParamsAPI":
        more_hidden_params = [
            "add_eos_token",
        ]
    elif class_name == "LLMGenericTrainingParamsAPI":
        more_hidden_params = [
            "model_ref",
            "dpo_beta",
            "max_prompt_length",
            "max_completion_length",
        ]
    elif class_name == "LLMPPOTrainingParamsAPI":
        more_hidden_params = [
            "dpo_beta",
            "max_prompt_length",
            "max_completion_length",
        ]
    else:
        more_hidden_params = []
    _excluded = FIELDS_TO_EXCLUDE + more_hidden_params
    new_fields: Dict[str, Tuple[Any, Any]] = {}
    for name, field in base_class.model_fields.items():
        if name not in _excluded:
            field_type = annotations[name]
            if field.default is not None:
                field_default = field.default
            elif field.default_factory is not None:
                field_default = field.default_factory
            else:
                field_default = None
            new_fields[name] = (field_type, field_default)
    config_dict = ConfigDict(protected_namespaces=())
    return create_model(
        class_name,
        __config__=config_dict,
        **{key: (value[0], value[1]) for key, value in new_fields.items()},
    )


LLMSFTTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMSFTTrainingParamsAPI")
LLMDPOTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMDPOTrainingParamsAPI")
LLMORPOTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMORPOTrainingParamsAPI")
LLMGenericTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMGenericTrainingParamsAPI")
LLMRewardTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMRewardTrainingParamsAPI")
LLMPPOTrainingParamsAPI = create_api_base_model(LLMTrainingParams, "LLMPPOTrainingParamsAPI")
ImageClassificationParamsAPI = create_api_base_model(ImageClassificationParams, "ImageClassificationParamsAPI")
Seq2SeqParamsAPI = create_api_base_model(Seq2SeqParams, "Seq2SeqParamsAPI")
TabularClassificationParamsAPI = create_api_base_model(TabularParams, "TabularClassificationParamsAPI")
TabularRegressionParamsAPI = create_api_base_model(TabularParams, "TabularRegressionParamsAPI")
TextClassificationParamsAPI = create_api_base_model(TextClassificationParams, "TextClassificationParamsAPI")
TextRegressionParamsAPI = create_api_base_model(TextRegressionParams, "TextRegressionParamsAPI")
TokenClassificationParamsAPI = create_api_base_model(TokenClassificationParams, "TokenClassificationParamsAPI")
SentenceTransformersParamsAPI = create_api_base_model(SentenceTransformersParams, "SentenceTransformersParamsAPI")
ImageRegressionParamsAPI = create_api_base_model(ImageRegressionParams, "ImageRegressionParamsAPI")
VLMTrainingParamsAPI = create_api_base_model(VLMTrainingParams, "VLMTrainingParamsAPI")
ExtractiveQuestionAnsweringParamsAPI = create_api_base_model(
    ExtractiveQuestionAnsweringParams, "ExtractiveQuestionAnsweringParamsAPI"
)
ObjectDetectionParamsAPI = create_api_base_model(ObjectDetectionParams, "ObjectDetectionParamsAPI")


class LLMSFTColumnMapping(BaseModel):
    text_column: str


class LLMDPOColumnMapping(BaseModel):
    text_column: str
    rejected_text_column: str
    prompt_text_column: str


class LLMORPOColumnMapping(BaseModel):
    text_column: str
    rejected_text_column: str
    prompt_text_column: str


class LLMGenericColumnMapping(BaseModel):
    text_column: str


class LLMRewardColumnMapping(BaseModel):
    text_column: str
    rejected_text_column: str


class ImageClassificationColumnMapping(BaseModel):
    image_column: str
    target_column: str


class ImageRegressionColumnMapping(BaseModel):
    image_column: str
    target_column: str


class Seq2SeqColumnMapping(BaseModel):
    text_column: str
    target_column: str


class TabularClassificationColumnMapping(BaseModel):
    id_column: str
    target_columns: List[str]


class TabularRegressionColumnMapping(BaseModel):
    id_column: str
    target_columns: List[str]


class TextClassificationColumnMapping(BaseModel):
    text_column: str
    target_column: str


class TextRegressionColumnMapping(BaseModel):
    text_column: str
    target_column: str


class TokenClassificationColumnMapping(BaseModel):
    tokens_column: str
    tags_column: str


class STPairColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str


class STPairClassColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str
    target_column: str


class STPairScoreColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str
    target_column: str


class STTripletColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str
    sentence3_column: str


class STQAColumnMapping(BaseModel):
    sentence1_column: str
    sentence2_column: str


class VLMColumnMapping(BaseModel):
    image_column: str
    text_column: str
    prompt_text_column: str


class ExtractiveQuestionAnsweringColumnMapping(BaseModel):
    text_column: str
    question_column: str
    answer_column: str


class ObjectDetectionColumnMapping(BaseModel):
    image_column: str
    objects_column: str


class APICreateProjectModel(BaseModel):
    project_name: str
    task: Literal[
        "llm:sft",
        "llm:dpo",
        "llm:orpo",
        "llm:generic",
        "llm:reward",
        "llm:ppo",
        "llm:distillation",
        "st:pair",
        "st:pair_class",
        "st:pair_score",
        "st:triplet",
        "st:qa",
        "image-classification",
        "seq2seq",
        "token-classification",
        "text-classification",
        "text-regression",
        "tabular-classification",
        "tabular-regression",
        "image-regression",
        "vlm:captioning",
        "vlm:vqa",
        "extractive-question-answering",
        "extractive-qa",
        "image-object-detection",
    ]
    base_model: str
    hardware: Literal[
        "spaces-a10g-large",
        "spaces-a10g-small",
        "spaces-a100-large",
        "spaces-t4-medium",
        "spaces-t4-small",
        "spaces-cpu-upgrade",
        "spaces-cpu-basic",
        "spaces-l4x1",
        "spaces-l4x4",
        "spaces-l40sx1",
        "spaces-l40sx4",
        "spaces-l40sx8",
        "spaces-a10g-largex2",
        "spaces-a10g-largex4",
        "local-ui",
    ]
    params: Union[
        LLMSFTTrainingParamsAPI,
        LLMDPOTrainingParamsAPI,
        LLMORPOTrainingParamsAPI,
        LLMGenericTrainingParamsAPI,
        LLMRewardTrainingParamsAPI,
        LLMPPOTrainingParamsAPI,
        SentenceTransformersParamsAPI,
        ImageClassificationParamsAPI,
        Seq2SeqParamsAPI,
        TabularClassificationParamsAPI,
        TabularRegressionParamsAPI,
        TextClassificationParamsAPI,
        TextRegressionParamsAPI,
        TokenClassificationParamsAPI,
        ImageRegressionParamsAPI,
        VLMTrainingParamsAPI,
        ExtractiveQuestionAnsweringParamsAPI,
        ObjectDetectionParamsAPI,
    ]
    username: str
    column_mapping: Optional[
        Union[
            LLMSFTColumnMapping,
            LLMDPOColumnMapping,
            LLMORPOColumnMapping,
            LLMGenericColumnMapping,
            LLMRewardColumnMapping,
            ImageClassificationColumnMapping,
            Seq2SeqColumnMapping,
            TabularClassificationColumnMapping,
            TabularRegressionColumnMapping,
            TextClassificationColumnMapping,
            TextRegressionColumnMapping,
            TokenClassificationColumnMapping,
            STPairColumnMapping,
            STPairClassColumnMapping,
            STPairScoreColumnMapping,
            STTripletColumnMapping,
            STQAColumnMapping,
            ImageRegressionColumnMapping,
            VLMColumnMapping,
            ExtractiveQuestionAnsweringColumnMapping,
            ObjectDetectionColumnMapping,
        ]
    ] = None
    hub_dataset: str
    train_split: str
    valid_split: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_column_mapping(cls, values):
        # Logic from original file kept for brevity...
        # (This logic is unchanged, so I'm simplifying the copy-paste if possible, but I must provide full content)
        # To avoid massive token usage I will include the logic.
        task = values.get("task")
        if task == "extractive-qa":
            values["task"] = "extractive-question-answering"
            task = "extractive-question-answering"

        # ... (All the validation logic)
        # Assuming I need to provide the full content, I will include it.
        # But wait, I can just replace the relevant parts if I use replace_in_file properly.
        # But the user asked for a "rewrite" implicitly by pointing out many issues.
        # I'll use write_to_file to ensure clean state.

        # I'll copy the validation logic from previous file content.
        if task == "llm:sft":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping is required for llm:sft")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column is required for llm:sft")
            values["column_mapping"] = LLMSFTColumnMapping(**values["column_mapping"])
        elif task == "llm:dpo":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            if not values.get("column_mapping").get("rejected_text_column"):
                raise HTTPException(status_code=422, detail="rejected_text_column required")
            if not values.get("column_mapping").get("prompt_text_column"):
                raise HTTPException(status_code=422, detail="prompt_text_column required")
            values["column_mapping"] = LLMDPOColumnMapping(**values["column_mapping"])
        elif task == "llm:orpo":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            if not values.get("column_mapping").get("rejected_text_column"):
                raise HTTPException(status_code=422, detail="rejected_text_column required")
            if not values.get("column_mapping").get("prompt_text_column"):
                raise HTTPException(status_code=422, detail="prompt_text_column required")
            values["column_mapping"] = LLMORPOColumnMapping(**values["column_mapping"])
        elif task == "llm:generic":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            values["column_mapping"] = LLMGenericColumnMapping(**values["column_mapping"])
        elif task == "llm:reward":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            if not values.get("column_mapping").get("rejected_text_column"):
                raise HTTPException(status_code=422, detail="rejected_text_column required")
            values["column_mapping"] = LLMRewardColumnMapping(**values["column_mapping"])
        elif task == "llm:ppo":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            values["column_mapping"] = LLMSFTColumnMapping(**values["column_mapping"])
        elif task == "llm:distillation":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            values["column_mapping"] = LLMSFTColumnMapping(**values["column_mapping"])
        elif task == "seq2seq":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            if not values.get("column_mapping").get("target_column"):
                raise HTTPException(status_code=422, detail="target_column required")
            values["column_mapping"] = Seq2SeqColumnMapping(**values["column_mapping"])
        elif task == "image-classification":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("image_column"):
                raise HTTPException(status_code=422, detail="image_column required")
            if not values.get("column_mapping").get("target_column"):
                raise HTTPException(status_code=422, detail="target_column required")
            values["column_mapping"] = ImageClassificationColumnMapping(**values["column_mapping"])
        elif task == "tabular-classification":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("id_column"):
                raise HTTPException(status_code=422, detail="id_column required")
            if not values.get("column_mapping").get("target_columns"):
                raise HTTPException(status_code=422, detail="target_columns required")
            values["column_mapping"] = TabularClassificationColumnMapping(**values["column_mapping"])
        elif task == "tabular-regression":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("id_column"):
                raise HTTPException(status_code=422, detail="id_column required")
            if not values.get("column_mapping").get("target_columns"):
                raise HTTPException(status_code=422, detail="target_columns required")
            values["column_mapping"] = TabularRegressionColumnMapping(**values["column_mapping"])
        elif task == "text-classification":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            if not values.get("column_mapping").get("target_column"):
                raise HTTPException(status_code=422, detail="target_column required")
            values["column_mapping"] = TextClassificationColumnMapping(**values["column_mapping"])
        elif task == "text-regression":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            if not values.get("column_mapping").get("target_column"):
                raise HTTPException(status_code=422, detail="target_column required")
            values["column_mapping"] = TextRegressionColumnMapping(**values["column_mapping"])
        elif task == "token-classification":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("tokens_column"):
                raise HTTPException(status_code=422, detail="tokens_column required")
            if not values.get("column_mapping").get("tags_column"):
                raise HTTPException(status_code=422, detail="tags_column required")
            values["column_mapping"] = TokenClassificationColumnMapping(**values["column_mapping"])
        elif task == "st:pair":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("sentence1_column"):
                raise HTTPException(status_code=422, detail="sentence1_column required")
            if not values.get("column_mapping").get("sentence2_column"):
                raise HTTPException(status_code=422, detail="sentence2_column required")
            values["column_mapping"] = STPairColumnMapping(**values["column_mapping"])
        elif task == "st:pair_class":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("sentence1_column"):
                raise HTTPException(status_code=422, detail="sentence1_column required")
            if not values.get("column_mapping").get("sentence2_column"):
                raise HTTPException(status_code=422, detail="sentence2_column required")
            if not values.get("column_mapping").get("target_column"):
                raise HTTPException(status_code=422, detail="target_column required")
            values["column_mapping"] = STPairClassColumnMapping(**values["column_mapping"])
        elif task == "st:pair_score":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("sentence1_column"):
                raise HTTPException(status_code=422, detail="sentence1_column required")
            if not values.get("column_mapping").get("sentence2_column"):
                raise HTTPException(status_code=422, detail="sentence2_column required")
            if not values.get("column_mapping").get("target_column"):
                raise HTTPException(status_code=422, detail="target_column required")
            values["column_mapping"] = STPairScoreColumnMapping(**values["column_mapping"])
        elif task == "st:triplet":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("sentence1_column"):
                raise HTTPException(status_code=422, detail="sentence1_column required")
            if not values.get("column_mapping").get("sentence2_column"):
                raise HTTPException(status_code=422, detail="sentence2_column required")
            if not values.get("column_mapping").get("sentence3_column"):
                raise HTTPException(status_code=422, detail="sentence3_column required")
            values["column_mapping"] = STTripletColumnMapping(**values["column_mapping"])
        elif task == "st:qa":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("sentence1_column"):
                raise HTTPException(status_code=422, detail="sentence1_column required")
            if not values.get("column_mapping").get("sentence2_column"):
                raise HTTPException(status_code=422, detail="sentence2_column required")
            values["column_mapping"] = STQAColumnMapping(**values["column_mapping"])
        elif task == "image-regression":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("image_column"):
                raise HTTPException(status_code=422, detail="image_column required")
            if not values.get("column_mapping").get("target_column"):
                raise HTTPException(status_code=422, detail="target_column required")
            values["column_mapping"] = ImageRegressionColumnMapping(**values["column_mapping"])
        elif task == "vlm:captioning":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("image_column"):
                raise HTTPException(status_code=422, detail="image_column required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            if not values.get("column_mapping").get("prompt_text_column"):
                raise HTTPException(status_code=422, detail="prompt_text_column required")
            values["column_mapping"] = VLMColumnMapping(**values["column_mapping"])
        elif task == "vlm:vqa":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("image_column"):
                raise HTTPException(status_code=422, detail="image_column required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            if not values.get("column_mapping").get("prompt_text_column"):
                raise HTTPException(status_code=422, detail="prompt_text_column required")
            values["column_mapping"] = VLMColumnMapping(**values["column_mapping"])
        elif task == "extractive-question-answering":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("text_column"):
                raise HTTPException(status_code=422, detail="text_column required")
            if not values.get("column_mapping").get("question_column"):
                raise HTTPException(status_code=422, detail="question_column required")
            if not values.get("column_mapping").get("answer_column"):
                raise HTTPException(status_code=422, detail="answer_column required")
            values["column_mapping"] = ExtractiveQuestionAnsweringColumnMapping(**values["column_mapping"])
        elif task == "image-object-detection":
            if not values.get("column_mapping"):
                raise HTTPException(status_code=422, detail="column_mapping required")
            if not values.get("column_mapping").get("image_column"):
                raise HTTPException(status_code=422, detail="image_column required")
            if not values.get("column_mapping").get("objects_column"):
                raise HTTPException(status_code=422, detail="objects_column required")
            values["column_mapping"] = ObjectDetectionColumnMapping(**values["column_mapping"])
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_params(cls, values):
        task = values.get("task")
        if not task:
            return values
        if task == "llm:sft":
            values["params"] = LLMSFTTrainingParamsAPI(**values["params"])
        elif task == "llm:dpo":
            values["params"] = LLMDPOTrainingParamsAPI(**values["params"])
        elif task == "llm:orpo":
            values["params"] = LLMORPOTrainingParamsAPI(**values["params"])
        elif task == "llm:generic":
            values["params"] = LLMGenericTrainingParamsAPI(**values["params"])
        elif task == "llm:reward":
            values["params"] = LLMRewardTrainingParamsAPI(**values["params"])
        elif task == "llm:ppo":
            values["params"] = LLMPPOTrainingParamsAPI(**values["params"])
        elif task == "llm:distillation":
            values["params"] = LLMSFTTrainingParamsAPI(**values["params"])
        elif task == "seq2seq":
            values["params"] = Seq2SeqParamsAPI(**values["params"])
        elif task == "image-classification":
            values["params"] = ImageClassificationParamsAPI(**values["params"])
        elif task == "tabular-classification":
            values["params"] = TabularClassificationParamsAPI(**values["params"])
        elif task == "tabular-regression":
            values["params"] = TabularRegressionParamsAPI(**values["params"])
        elif task == "text-classification":
            values["params"] = TextClassificationParamsAPI(**values["params"])
        elif task == "text-regression":
            values["params"] = TextRegressionParamsAPI(**values["params"])
        elif task == "token-classification":
            values["params"] = TokenClassificationParamsAPI(**values["params"])
        elif task.startswith("st:"):
            values["params"] = SentenceTransformersParamsAPI(**values["params"])
        elif task == "image-regression":
            values["params"] = ImageRegressionParamsAPI(**values["params"])
        elif task.startswith("vlm:"):
            values["params"] = VLMTrainingParamsAPI(**values["params"])
        elif task == "extractive-question-answering":
            values["params"] = ExtractiveQuestionAnsweringParamsAPI(**values["params"])
        elif task == "image-object-detection":
            values["params"] = ObjectDetectionParamsAPI(**values["params"])
        return values


class JobIDModel(BaseModel):
    jid: str


api_router = APIRouter()


def api_auth(request: Request):
    # Allow local access without token if env var is not set
    # This assumes secure local environment if HF_TOKEN is missing
    if os.environ.get("HF_TOKEN") is None:
        return True

    authorization = request.headers.get("Authorization")
    if authorization:
        schema, _, token = authorization.partition(" ")
        if schema.lower() == "bearer":
            token = token.strip()
            try:
                _ = token_verification(token=token)
                return token
            except Exception as e:
                logger.error(f"Failed to verify token: {e}")
                raise HTTPException(status_code=401, detail="Invalid or expired token: Bearer")
    raise HTTPException(status_code=401, detail="Invalid or expired token")


@api_router.post("/create_project", response_class=JSONResponse)
async def api_create_project(project: APICreateProjectModel, token: bool = Depends(api_auth)):
    provided_params = project.params.model_dump()
    hardware = "local-ui" if project.hardware == "local" else project.hardware

    logger.info(provided_params)
    logger.info(project.column_mapping)

    task = project.task
    if task.startswith("llm"):
        params = PARAMS["llm"]
        trainer = task.split(":")[1]
        params.update({"trainer": trainer})
    elif task.startswith("st:"):
        params = PARAMS["st"]
        trainer = task.split(":")[1]
        params.update({"trainer": trainer})
    elif task.startswith("vlm:"):
        params = PARAMS["vlm"]
        trainer = task.split(":")[1]
        params.update({"trainer": trainer})
    elif task.startswith("tabular"):
        params = PARAMS["tabular"]
    else:
        params = PARAMS[task]

    params.update(provided_params)

    app_params = AppParams(
        job_params_json=json.dumps(params),
        token=token,
        project_name=project.project_name,
        username=project.username,
        task=task,
        data_path=project.hub_dataset,
        base_model=project.base_model,
        column_mapping=project.column_mapping.model_dump() if project.column_mapping else None,
        using_hub_dataset=True,
        train_split=project.train_split,
        valid_split=project.valid_split,
        api=True,
    )
    params = app_params.munge()
    project = AutoTrainProject(params=params, backend=hardware)
    job_id = project.create()
    return {"message": "Project created", "job_id": job_id, "success": True}


@api_router.get("/version", response_class=JSONResponse)
async def api_version():
    return {"version": __version__}


@api_router.post("/stop_training", response_class=JSONResponse)
async def api_stop_training(job: JobIDModel, token: bool = Depends(api_auth)):
    hf_api = HfApi(token=token)
    job_id = job.jid
    try:
        hf_api.pause_space(repo_id=job_id)
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        return {"message": f"Failed to stop training for {job_id}: {e}", "success": False}
    return {"message": f"Training stopped for {job_id}", "success": True}


@api_router.post("/logs", response_class=JSONResponse)
async def api_logs(job: JobIDModel, token: bool = Depends(api_auth)):
    job_id = job.jid
    jwt_url = f"{constants.ENDPOINT}/api/spaces/{job_id}/jwt"
    response = get_session().get(jwt_url, headers=build_hf_headers(token=token))
    hf_raise_for_status(response)
    jwt_token = response.json()["token"]

    logs_url = f"https://api.hf.space/v1/{job_id}/logs/run"

    _logs = []
    try:
        with get_session().get(
            logs_url, headers=build_hf_headers(token=jwt_token), stream=True, timeout=3
        ) as response:
            hf_raise_for_status(response)
            for line in response.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                line_data = line[len(b"data: ") :]
                try:
                    event = json.loads(line_data.decode())
                except json.JSONDecodeError:
                    continue
                _logs.append((event["timestamp"], event["data"]))

        _logs = "\n".join([f"{timestamp}: {data}" for timestamp, data in _logs])
        return {"logs": _logs, "success": True, "message": "Logs fetched successfully"}
    except Exception as e:
        if "Read timed out" in str(e):
            _logs = "\n".join([f"{timestamp}: {data}" for timestamp, data in _logs])
            return {"logs": _logs, "success": True, "message": "Logs fetched successfully"}
        return {"logs": str(e), "success": False, "message": "Failed to fetch logs"}


# --- Inference & Discovery ---


def get_models_dir() -> str:
    """Get the allowed root directory for models.

    Uses the same logic as project normalization during training:
    - AITRAINING_MODELS_DIR if set (primary)
    - AUTOTRAIN_MODELS_DIR if set (backward compatibility)
    - AUTOTRAIN_PROJECTS_DIR if set
    - Otherwise ../trainings/ directory (where models are saved by default)
    """
    # Check for new environment variable first
    models_dir = os.environ.get("AITRAINING_MODELS_DIR")
    if models_dir:
        return models_dir

    # Check for old variable for backward compatibility
    models_dir = os.environ.get("AUTOTRAIN_MODELS_DIR")
    if models_dir:
        return models_dir

    # Use the same logic as training for consistency
    projects_dir = os.environ.get("AUTOTRAIN_PROJECTS_DIR")
    if projects_dir:
        return projects_dir

    # Default to ../trainings/ where models are saved by default
    server_parent = os.path.dirname(os.getcwd())
    return os.path.join(server_parent, "trainings")


def validate_model_path(model_id: str) -> str:
    """
    Validate and resolve the model path.
    Supports both local models and HuggingFace Hub model IDs.

    For local models: prevent directory traversal and ensure path is within allowed directory.
    For HF Hub models: return the model_id as-is to be downloaded by transformers/sentence-transformers.
    """
    if not model_id or ".." in model_id:
        raise HTTPException(status_code=400, detail="Invalid model_id")

    # Allow forward slashes for project/model structure but not backslashes
    if "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model_id")

    # First, try to resolve as a local model
    root_dir = os.path.abspath(get_models_dir())
    model_path = os.path.abspath(os.path.join(root_dir, model_id))

    # If it's a local model and within the allowed directory, validate and return
    if model_path.startswith(root_dir) and os.path.exists(model_path):
        return model_path

    # Otherwise, assume it's a HuggingFace Hub model ID
    # Valid HF model IDs are in the format: "username/model-name" or just "model-name"
    # They should not contain path-like elements beyond a single slash
    if model_id.count("/") > 1:
        raise HTTPException(status_code=400, detail="Invalid model_id format")

    # Return the HF Hub model ID as-is - transformers will download it
    logger.info(f"Model not found locally, assuming HuggingFace Hub model: {model_id}")
    return model_id


@api_router.get("/models/list", response_class=JSONResponse)
async def api_models_list(token: bool = Depends(api_auth)):
    models = []
    root_dir = get_models_dir()

    try:
        if not os.path.exists(root_dir):
            return []

        for item in os.listdir(root_dir):
            if item.startswith(".") or item in ("node_modules", "__pycache__", "src", "docs", "tests"):
                continue

            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                # Check if this is a training project directory with a model subdirectory
                model_subdir = os.path.join(item_path, "model")
                if os.path.exists(model_subdir) and os.path.isdir(model_subdir):
                    # The actual model is in the model/ subdirectory
                    model_type = detect_model_type(model_subdir)
                    if model_type != "unknown":
                        metadata = get_model_metadata(model_subdir)
                        models.append(
                            {
                                "id": f"{item}/model",  # Include /model in the ID
                                "type": model_type,
                                "model_path": model_subdir,
                                "metadata": metadata,
                            }
                        )
                else:
                    # Try the directory itself (backward compatibility)
                    model_type = detect_model_type(item_path)
                    if model_type != "unknown":
                        metadata = get_model_metadata(item_path)
                        models.append({"id": item, "type": model_type, "model_path": item_path, "metadata": metadata})
    except Exception as e:
        logger.error(f"Error scanning for models: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    return models


class UniversalInferenceRequest(BaseModel):
    model_id: str
    inputs: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    task_override: Optional[str] = None  # Optionally override auto-detected model type


def get_cached_pipeline(model_path: str, task: str, device="cpu", token: str = None):
    key = (model_path, task)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    logger.info(f"Loading model {model_path} for task {task}")
    pipe = pipeline(task, model=model_path, device=device, token=token)
    MODEL_CACHE[key] = pipe
    return pipe


def get_cached_vlm(model_path: str, device="cpu"):
    key = (model_path, "vlm")
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    logger.info(f"Loading VLM {model_path}")

    # Check if this is a PEFT model
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        # This is a PEFT model, load it properly
        try:
            import json

            from peft import PeftModel

            # Read adapter config to get base model
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")

            # Load base model first
            try:
                base_model = AutoModelForVision2Seq.from_pretrained(base_model_name, local_files_only=True)
            except Exception:
                base_model = AutoModelForVision2Seq.from_pretrained(base_model_name)

            # Load PEFT adapters
            model = PeftModel.from_pretrained(base_model, model_path)
        except ImportError:
            raise ImportError("PEFT library required for adapter models. Install with: pip install peft")
    else:
        # Regular model
        model = AutoModelForVision2Seq.from_pretrained(model_path)

    processor = AutoProcessor.from_pretrained(model_path)
    if device == "cuda":
        model = model.to("cuda")

    MODEL_CACHE[key] = (model, processor)
    return model, processor


def get_cached_llm(model_path: str, config=None):
    """
    Get cached LLM completer for a model.

    The completer (with model/tokenizer) is cached, but config can be updated
    per request for dynamic parameter changes without reloading the model.
    """
    from autotrain.generation import CompletionConfig, create_completer

    key = (model_path, "llm")
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    # Create default config if not provided (will be updated per request)
    if config is None:
        config = CompletionConfig()

    completer = create_completer(model=model_path, tokenizer=model_path, completer_type="message", config=config)

    # Model is already moved to the correct device by create_completer
    # No need to move again - that's actually SLOW because it copies the model

    MODEL_CACHE[key] = completer
    return completer


@api_router.post("/inference/universal")
async def universal_inference(
    request: UniversalInferenceRequest, token: bool = Depends(api_auth), authorization: str = Header(None)
):
    try:
        # Extract HF token from Authorization header if provided
        hf_token = None
        if authorization and authorization.startswith("Bearer "):
            hf_token = authorization.replace("Bearer ", "")

        model_path = validate_model_path(request.model_id)
        # Use task_override if provided, otherwise auto-detect
        model_type = request.task_override if request.task_override else detect_model_type(model_path)

        # Detect best available device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Using device: {device}")

        if model_type == "llm":
            pass

            text = request.inputs.get("text")
            if not text:
                raise HTTPException(status_code=400, detail="Input 'text' required")

            params = request.parameters or {}

            # Cache model/tokenizer separately, not the full completer
            # This allows us to use different generation params per request
            completer = get_cached_llm(model_path, config=None)

            # Update completer config with request parameters
            # This is much faster than recreating the completer
            completer.config.max_new_tokens = params.get("max_new_tokens", 100)
            completer.config.temperature = params.get("temperature", 0.7)
            completer.config.top_p = params.get("top_p", 0.95)
            completer.config.top_k = params.get("top_k", 50)
            completer.config.do_sample = params.get("do_sample", True)

            # Build conversation with system prompt if provided
            system_prompt = request.inputs.get("system_prompt")
            conversation = [{"role": "user", "content": text}]

            result = completer.complete(conversation, system_prompt=system_prompt)

            return {"outputs": [result.text], "model_type": "llm"}

        elif model_type in ["text-classification", "token-classification", "text-regression"]:
            pipe = get_cached_pipeline(model_path, model_type, device, hf_token)
            text = request.inputs.get("text")
            if not text:
                raise HTTPException(status_code=400, detail="Input 'text' required")
            output = pipe(text)
            return {"outputs": json.loads(json.dumps(output, default=str)), "model_type": model_type}

        elif model_type in ["image-classification", "image-object-detection"]:
            image_data = request.inputs.get("image")
            if not image_data:
                raise HTTPException(status_code=400, detail="Input 'image' required")

            if len(image_data) > MAX_IMAGE_SIZE:
                raise HTTPException(status_code=400, detail="Image too large")

            if "," in image_data:
                image_data = image_data.split(",")[1]

            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image data")

            task = "object-detection" if model_type == "image-object-detection" else "image-classification"
            pipe = get_cached_pipeline(model_path, task, device, hf_token)
            output = pipe(image)
            return {"outputs": output, "model_type": model_type}

        elif model_type == "seq2seq":
            # Seq2seq models (T5, BART, etc.) - text-to-text generation
            text = request.inputs.get("text")
            if not text:
                raise HTTPException(status_code=400, detail="Input 'text' required")

            pipe = get_cached_pipeline(model_path, "text2text-generation", device, hf_token)

            # Get generation parameters
            params = request.parameters or {}
            gen_kwargs = {
                "max_new_tokens": params.get("max_new_tokens", 100),
                "temperature": params.get("temperature", 1.0),
                "top_p": params.get("top_p", 1.0),
                "top_k": params.get("top_k", 50),
                "do_sample": params.get("do_sample", False),
            }

            output = pipe(text, **gen_kwargs)
            return {"outputs": [output[0]["generated_text"]], "model_type": "seq2seq"}

        elif model_type == "vlm":
            image_data = request.inputs.get("image")
            text = request.inputs.get("text", "Describe this image")

            if not image_data:
                raise HTTPException(status_code=400, detail="Input 'image' required")
            if len(image_data) > MAX_IMAGE_SIZE:
                raise HTTPException(status_code=400, detail="Image too large")

            if "," in image_data:
                image_data = image_data.split(",")[1]

            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image data")

            model, processor = get_cached_vlm(model_path, device)

            inputs = processor(text=text, images=image, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            generated_ids = model.generate(**inputs, max_new_tokens=50)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return {"outputs": [generated_text], "model_type": "vlm"}

        elif model_type == "extractive-question-answering":
            # Question answering models
            text = request.inputs.get("text")  # context
            question = request.inputs.get("question")
            if not text or not question:
                raise HTTPException(status_code=400, detail="Both 'text' (context) and 'question' required")

            pipe = get_cached_pipeline(model_path, "question-answering", device, hf_token)
            output = pipe(question=question, context=text)
            return {"outputs": [output["answer"]], "model_type": "extractive-question-answering"}

        elif model_type == "sentence-transformers":
            # Sentence transformers - generate embeddings
            texts = request.inputs.get("texts")  # Can be single text or list
            if not texts:
                raise HTTPException(status_code=400, detail="Input 'texts' required (string or list of strings)")

            # Normalize to list
            if isinstance(texts, str):
                texts = [texts]

            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_path, token=hf_token)
            if device == "cuda":
                model = model.to("cuda")

            embeddings = model.encode(texts)
            return {"outputs": embeddings.tolist(), "model_type": "sentence-transformers"}

        elif model_type == "image-regression":
            # Image regression - predict numerical values from images
            image_data = request.inputs.get("image")
            if not image_data:
                raise HTTPException(status_code=400, detail="Input 'image' required")

            if len(image_data) > MAX_IMAGE_SIZE:
                raise HTTPException(status_code=400, detail="Image too large")

            if "," in image_data:
                image_data = image_data.split(",")[1]

            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image data")

            pipe = get_cached_pipeline(model_path, "image-classification", device, hf_token)  # Uses same pipeline
            output = pipe(image)

            # For regression, extract the predicted value
            if isinstance(output, list) and len(output) > 0:
                # Assuming top prediction is the regression value
                predicted_value = output[0].get("score", 0.0)
                return {"outputs": [predicted_value], "model_type": "image-regression"}
            else:
                return {"outputs": output, "model_type": "image-regression"}

        elif model_type == "tabular":
            # Tabular models - structured data prediction
            features = request.inputs.get("features")
            if not features:
                raise HTTPException(status_code=400, detail="Input 'features' required (dict of feature values)")

            # Tabular models typically use custom inference, not HF pipelines
            # This is a simplified version - real implementation depends on how the model was trained
            import pandas as pd
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            if device == "cuda":
                model = model.to("cuda")

            # Convert features dict to DataFrame
            pd.DataFrame([features])

            # This is simplified - real tabular models may need custom preprocessing
            # For now, just return a placeholder
            return {"outputs": ["Tabular inference not fully implemented yet"], "model_type": "tabular"}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/conversations")
async def get_conversations(model_id: str, token: bool = Depends(api_auth)):
    try:
        logger.info(f"Getting conversations for model_id: {model_id}")
        model_path = validate_model_path(model_id)
        logger.info(f"Model path resolved to: {model_path}")
        conv_dir = os.path.join(model_path, "conversations")
        logger.info(f"Looking for conversations in: {conv_dir}")

        if not os.path.exists(conv_dir):
            logger.info(f"Conversations directory does not exist: {conv_dir}")
            return []

        conversations = []
        for f in os.listdir(conv_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(conv_dir, f)) as file:
                        conversations.append(json.load(file))
                except Exception:
                    pass

        conversations.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return conversations
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return []


@api_router.post("/conversations/save")
async def save_conversation(model_id: str, conversation: Dict[str, Any], token: bool = Depends(api_auth)):
    try:
        logger.info(f"Saving conversation for model_id: {model_id}")
        model_path = validate_model_path(model_id)
        logger.info(f"Model path resolved to: {model_path}")
        conv_dir = os.path.join(model_path, "conversations")
        os.makedirs(conv_dir, exist_ok=True)
        logger.info(f"Created/verified conversations directory: {conv_dir}")

        timestamp = conversation.get("timestamp", int(time.time()))
        conversation_id = str(timestamp)
        filename = f"{conversation_id}.json"

        # Validate filename
        if not conversation_id.replace(".", "").isdigit():
            raise HTTPException(status_code=400, detail="Invalid timestamp")

        with open(os.path.join(conv_dir, filename), "w") as f:
            json.dump(conversation, f, indent=2)

        return {"success": True, "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class InferenceRequest(BaseModel):
    model_path: str
    prompts: List[str]
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    do_sample: Optional[bool] = True
    device: Optional[str] = None


class InferenceResponse(BaseModel):
    outputs: List[str]
    model_path: str
    num_prompts: int


@api_router.post("/llm/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, token: bool = Depends(api_auth)):
    # kept for backward compatibility but using same logic structure could be improved
    # For now keeping as is but users should prefer /inference/universal
    try:
        from autotrain.generation import CompletionConfig, create_completer

        config = CompletionConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
        )
        completer = create_completer(model=request.model_path, completer_type="token", config=config)
        if request.device:
            import torch

            device = torch.device(request.device)
            completer.model = completer.model.to(device)
        outputs = []
        for prompt in request.prompts:
            result = completer.complete(prompt)
            outputs.append(result.text)
        return InferenceResponse(outputs=outputs, model_path=request.model_path, num_prompts=len(request.prompts))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
