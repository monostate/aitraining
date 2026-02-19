import os
import shlex
import sys

from autotrain import logger
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.generic.params import GenericParams
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


CPU_COMMAND = [
    sys.executable,
    "-m",
    "accelerate.commands.launch",
    "--cpu",
]

SINGLE_GPU_COMMAND = [
    sys.executable,
    "-m",
    "accelerate.commands.launch",
    "--num_machines",
    "1",
    "--num_processes",
    "1",
]


def get_accelerate_command(num_gpus, gradient_accumulation_steps=1, distributed_backend=None, ddp_timeout=None):
    """
    Generates the appropriate command to launch a training job using the `accelerate` library based on the number of GPUs
    and the specified distributed backend.

    Args:
        num_gpus (int): The number of GPUs available for training. If 0, training will be forced on CPU.
        gradient_accumulation_steps (int, optional): The number of gradient accumulation steps. Defaults to 1.
        distributed_backend (str, optional): The distributed backend to use. Can be "ddp" (Distributed Data Parallel),
                                             "deepspeed", or None. Defaults to None.

    Returns:
        list or str: The command to be executed as a list of strings. If no GPU is found, returns a CPU command string.
                     If a single GPU is found, returns a single GPU command string. Otherwise, returns a list of
                     command arguments for multi-GPU or DeepSpeed training.

    Raises:
        ValueError: If an unsupported distributed backend is specified.
    """
    if num_gpus == 0:
        logger.warning("No GPU found. Forcing training on CPU. This will be super slow!")
        return list(CPU_COMMAND)  # Return a copy to avoid mutation

    if num_gpus == 1:
        return list(SINGLE_GPU_COMMAND)  # Return a copy to avoid mutation

    if distributed_backend in ("ddp", None):
        cmd = [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_machines",
            "1",
            "--num_processes",
            str(num_gpus),
        ]
        if ddp_timeout is not None:
            cmd.extend(["--timeout", str(ddp_timeout)])
        return cmd
    elif distributed_backend == "deepspeed":
        cmd = [
            "accelerate",
            "launch",
            "--use_deepspeed",
            "--zero_stage",
            "3",
            "--offload_optimizer_device",
            "none",
            "--offload_param_device",
            "none",
            "--zero3_save_16bit_model",
            "true",
            "--zero3_init_flag",
            "true",
            "--deepspeed_multinode_launcher",
            "standard",
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
        ]
        if ddp_timeout is not None:
            cmd.extend(["--timeout", str(ddp_timeout)])
        return cmd
    else:
        raise ValueError("Unsupported distributed backend")


def launch_command(params):
    """
    Launches the appropriate training command based on the type of training parameters provided.

    Args:
        params (object): An instance of one of the training parameter classes. This can be one of the following:
            - LLMTrainingParams
            - GenericParams
            - TabularParams
            - TextClassificationParams
            - TextRegressionParams
            - SentenceTransformersParams
            - ExtractiveQuestionAnsweringParams
            - TokenClassificationParams
            - ImageClassificationParams
            - ObjectDetectionParams
            - ImageRegressionParams
            - Seq2SeqParams
            - VLMTrainingParams

    Returns:
        list: A list of command line arguments to be executed for training.

    Raises:
        ValueError: If the provided params type is unsupported.
    """

    params.project_name = shlex.split(params.project_name)[0]
    # Allow forcing GPU count to avoid importing torch (which can initialize CUDA in parent process)
    forced_num_gpus = os.environ.get("AUTOTRAIN_FORCE_NUM_GPUS")
    if forced_num_gpus is not None:
        try:
            num_gpus = int(forced_num_gpus)
        except Exception:
            num_gpus = 1
        cuda_available = num_gpus > 0
        mps_available = False
    else:
        # Import torch lazily to avoid CUDA initialization in the parent when not needed
        import torch  # type: ignore

        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        if cuda_available:
            num_gpus = torch.cuda.device_count()
        elif mps_available:
            # MPS has compatibility issues with quantized models and PEFT
            # Check if we should disable MPS and fall back to CPU
            should_disable_mps = False

            # Check for quantization which is known to cause issues with MPS
            if isinstance(params, (LLMTrainingParams, VLMTrainingParams, Seq2SeqParams)):
                if params.quantization and params.quantization != "none":
                    logger.warning(
                        f"Quantization ({params.quantization}) is not fully compatible with MPS. "
                        "Falling back to CPU training. Set AUTOTRAIN_ENABLE_MPS=1 to force MPS anyway."
                    )
                    should_disable_mps = True

            # Allow users to force MPS or CPU via environment variable
            force_mps = os.environ.get("AUTOTRAIN_ENABLE_MPS", "0") == "1"
            force_cpu = os.environ.get("AUTOTRAIN_DISABLE_MPS", "0") == "1"

            if force_cpu or (should_disable_mps and not force_mps):
                # Disable MPS by setting environment variables for subprocess
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                os.environ["AUTOTRAIN_DISABLE_MPS"] = "1"
                num_gpus = 0
                mps_available = False
            else:
                num_gpus = 1
        else:
            num_gpus = 0
    if isinstance(params, LLMTrainingParams):
        # Set NCCL timeout env var for long-running operations (e.g. GRPO reward scoring)
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = str(params.ddp_timeout)

        # For vLLM server mode, reserve GPUs for the vLLM server
        effective_num_gpus = num_gpus
        if getattr(params, "use_vllm", False) and getattr(params, "vllm_mode", "") == "server":
            effective_num_gpus = max(1, num_gpus - params.vllm_server_gpus)
            logger.info(
                f"vLLM server mode: reserving {params.vllm_server_gpus} GPU(s) for vLLM, "
                f"{effective_num_gpus} GPU(s) for training"
            )

        # Create a fresh copy of the command list to avoid accumulation
        cmd = list(get_accelerate_command(effective_num_gpus, params.gradient_accumulation, params.distributed_backend, ddp_timeout=params.ddp_timeout))
        if num_gpus > 0:
            cmd.append("--mixed_precision")
            if params.mixed_precision == "fp16":
                cmd.append("fp16")
            elif params.mixed_precision == "bf16":
                cmd.append("bf16")
            else:
                cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.clm",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )

    elif isinstance(params, GenericParams):
        cmd = [
            "python",
            "-m",
            "autotrain.trainers.generic",
            "--config",
            os.path.join(params.project_name, "training_params.json"),
        ]
    elif isinstance(params, TabularParams):
        cmd = [
            "python",
            "-m",
            "autotrain.trainers.tabular",
            "--training_config",
            os.path.join(params.project_name, "training_params.json"),
        ]
    elif (
        isinstance(params, TextClassificationParams)
        or isinstance(params, TextRegressionParams)
        or isinstance(params, SentenceTransformersParams)
        or isinstance(params, ExtractiveQuestionAnsweringParams)
    ):
        if num_gpus == 0:
            cmd = [
                "accelerate",
                "launch",
                "--cpu",
            ]
        elif num_gpus == 1:
            cmd = [
                "accelerate",
                "launch",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
            ]
        else:
            cmd = [
                "accelerate",
                "launch",
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                str(num_gpus),
            ]

        if num_gpus > 0:
            cmd.append("--mixed_precision")
            if params.mixed_precision == "fp16":
                cmd.append("fp16")
            elif params.mixed_precision == "bf16":
                cmd.append("bf16")
            else:
                cmd.append("no")

        if isinstance(params, TextRegressionParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.text_regression",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        elif isinstance(params, SentenceTransformersParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.sent_transformers",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        elif isinstance(params, ExtractiveQuestionAnsweringParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.extractive_question_answering",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        else:
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.text_classification",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
    elif isinstance(params, TokenClassificationParams):
        if num_gpus == 0:
            cmd = [
                "accelerate",
                "launch",
                "--cpu",
            ]
        elif num_gpus == 1:
            cmd = [
                "accelerate",
                "launch",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
            ]
        else:
            cmd = [
                "accelerate",
                "launch",
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                str(num_gpus),
            ]

        if num_gpus > 0:
            cmd.append("--mixed_precision")
            if params.mixed_precision == "fp16":
                cmd.append("fp16")
            elif params.mixed_precision == "bf16":
                cmd.append("bf16")
            else:
                cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.token_classification",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )
    elif (
        isinstance(params, ImageClassificationParams)
        or isinstance(params, ObjectDetectionParams)
        or isinstance(params, ImageRegressionParams)
    ):
        if num_gpus == 0:
            cmd = [
                "accelerate",
                "launch",
                "--cpu",
            ]
        elif num_gpus == 1:
            cmd = [
                "accelerate",
                "launch",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
            ]
        else:
            cmd = [
                "accelerate",
                "launch",
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                str(num_gpus),
            ]

        if num_gpus > 0:
            cmd.append("--mixed_precision")
            if params.mixed_precision == "fp16":
                cmd.append("fp16")
            elif params.mixed_precision == "bf16":
                cmd.append("bf16")
            else:
                cmd.append("no")

        if isinstance(params, ObjectDetectionParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.object_detection",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        elif isinstance(params, ImageRegressionParams):
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.image_regression",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
        else:
            cmd.extend(
                [
                    "-m",
                    "autotrain.trainers.image_classification",
                    "--training_config",
                    os.path.join(params.project_name, "training_params.json"),
                ]
            )
    elif isinstance(params, Seq2SeqParams):
        if num_gpus == 0:
            logger.warning("No GPU found. Forcing training on CPU. This will be super slow!")
            cmd = [
                "accelerate",
                "launch",
                "--cpu",
            ]
        elif num_gpus == 1:
            cmd = [
                "accelerate",
                "launch",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
            ]
        elif num_gpus == 2:
            cmd = [
                "accelerate",
                "launch",
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                "2",
            ]
        else:
            if params.quantization in ("int8", "int4") and params.peft and params.mixed_precision == "bf16":
                cmd = [
                    "accelerate",
                    "launch",
                    "--multi_gpu",
                    "--num_machines",
                    "1",
                    "--num_processes",
                    str(num_gpus),
                ]
            else:
                cmd = [
                    "accelerate",
                    "launch",
                    "--use_deepspeed",
                    "--zero_stage",
                    "3",
                    "--offload_optimizer_device",
                    "none",
                    "--offload_param_device",
                    "none",
                    "--zero3_save_16bit_model",
                    "true",
                    "--zero3_init_flag",
                    "true",
                    "--deepspeed_multinode_launcher",
                    "standard",
                    "--gradient_accumulation_steps",
                    str(params.gradient_accumulation),
                ]
        if num_gpus > 0:
            cmd.append("--mixed_precision")
            if params.mixed_precision == "fp16":
                cmd.append("fp16")
            elif params.mixed_precision == "bf16":
                cmd.append("bf16")
            else:
                cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.seq2seq",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )

    elif isinstance(params, VLMTrainingParams):
        if num_gpus == 0:
            logger.warning("No GPU found. Forcing training on CPU. This will be super slow!")
            cmd = [
                "accelerate",
                "launch",
                "--cpu",
            ]
        elif num_gpus == 1:
            cmd = [
                "accelerate",
                "launch",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
            ]
        elif num_gpus == 2:
            cmd = [
                "accelerate",
                "launch",
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                "2",
            ]
        else:
            if params.quantization in ("int8", "int4") and params.peft and params.mixed_precision == "bf16":
                cmd = [
                    "accelerate",
                    "launch",
                    "--multi_gpu",
                    "--num_machines",
                    "1",
                    "--num_processes",
                    str(num_gpus),
                ]
            else:
                cmd = [
                    "accelerate",
                    "launch",
                    "--use_deepspeed",
                    "--zero_stage",
                    "3",
                    "--offload_optimizer_device",
                    "none",
                    "--offload_param_device",
                    "none",
                    "--zero3_save_16bit_model",
                    "true",
                    "--zero3_init_flag",
                    "true",
                    "--deepspeed_multinode_launcher",
                    "standard",
                    "--gradient_accumulation_steps",
                    str(params.gradient_accumulation),
                ]

        if num_gpus > 0:
            cmd.append("--mixed_precision")
            if params.mixed_precision == "fp16":
                cmd.append("fp16")
            elif params.mixed_precision == "bf16":
                cmd.append("bf16")
            else:
                cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.vlm",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )

    else:
        raise ValueError("Unsupported params type")

    logger.info(cmd)
    logger.info(params)
    return cmd
