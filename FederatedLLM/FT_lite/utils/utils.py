from hydra import compose, initialize
from flwr_datasets import FederatedDataset
import matplotlib.pyplot as plt
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from omegaconf import DictConfig
import torch
import math
from peft.utils import prepare_model_for_kbit_training
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import flwr as fl
from flwr.common.typing import NDArrays, Scalar
from typing import Callable, Dict, Tuple
from collections import OrderedDict
from flwr.common import Context
from logging import WARNING, ERROR, LogRecord

########### to filter out all warnigns from HF coming from client side #########
backend_setup = {"logging_level": ERROR, "log_to_driver": False}


################################ configs components #############################

def get_config(config_name: str):
    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name=config_name)

    return cfg


############################# visualize data partitions #######################
def visualize_partitions(fed_dataset: FederatedDataset):
    _ = fed_dataset.load_partition(0)
    num_partitions = fed_dataset.partitioners['train'].num_partitions
    
    plt.bar(range(num_partitions), [len(fed_dataset.load_partition(i)) for i in range(num_partitions)])
    plt.xticks(range(num_partitions))
    plt.xlabel("Partition ID")
    plt.ylabel("Number of examples")
    plt.title(f"IID partitioning into {num_partitions} partitions")

################################ dataset components #############################


def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts

def get_tokenizer_and_data_collator_and_propt_formatting(
    model_name: str, use_fast: bool, padding_side: str
):

    # From: https://huggingface.co/docs/trl/en/sft_trainer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=use_fast, padding_side=padding_side
    )

    tokenizer.pad_token = (
        tokenizer.bos_token if padding_side == "left" else tokenizer.eos_token
    )
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func

################################ model components #############################


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropiate quantization config and
    other optimizations."""

    use_cuda = torch.cuda.is_available()
    quantization_config = None
    model_name = model_cfg.name
    if use_cuda:
        if model_cfg.quantization == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif model_cfg.quantization == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(
                f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
            )

        model_name = model_cfg.name

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if use_cuda:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )

    target_modules = model_cfg.lora.target_modules
    if target_modules:
        target_modules = list(target_modules)
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, peft_config)
    if not (use_cuda):
        peft_model.enable_input_require_grads()

    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False

    return peft_model

################################ client components #############################
# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        save_path,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.save_path = save_path

        # instantiate model
        self.model = get_model(model_cfg)

        self.trainset = trainset

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.train_cfg.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = self.save_path

        evalset = None
        if self.train_cfg.evaluate_split:
            train_test = self.trainset.train_test_split(test_size=0.1, seed=1234)
            trainset = train_test['train']
            evalset = train_test['test']
        else:
            trainset = self.trainset

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=trainset,
            eval_dataset=evalset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        metrics = {}
        if self.train_cfg.evaluate_split:
            eval_res = trainer.evaluate()
            metrics['eval_loss'] = eval_res['eval_loss']
            print(eval_res)

        # Do local training
        results = trainer.train()

        metrics = {**metrics, "train_loss": results.training_loss}

        return (
            self.get_parameters({}),
            len(self.trainset),
            metrics,
        )


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)

# Get a function that will be used to construct the config that the client's
# fit() method will receive
def get_on_fit_config():
    def fit_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        return fit_config

    return fit_config_fn



def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


################################ server components #############################


# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return 0.0, {}

    return evaluate


def load_pretrained_model(model_name: str = "EleutherAI/pythia-70m", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Carga el modelo preentrenado de Hugging Face y el tokenizador correspondiente.

    Parámetros:
        model_name (str): Nombre del modelo preentrenado (por defecto "EleutherAI/pythia-70m").
        device (str): Dispositivo para cargar el modelo ("cuda" o "cpu").

    Retorna:
        model (AutoModelForCausalLM): Modelo preentrenado cargado.
        tokenizer (AutoTokenizer): Tokenizador correspondiente al modelo.
    """
    # Cargar el tokenizador
    print(f"Cargando el tokenizador para {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configurar el token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cargar el modelo preentrenado
    print(f"Cargando el modelo preentrenado {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Mover el modelo al dispositivo especificado
    model.to(device)

    print(f"El modelo ha sido cargado exitosamente en el dispositivo: {device}")
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    """
    Genera texto utilizando un modelo de lenguaje preentrenado.

    Parámetros:
        model: Modelo preentrenado (e.g., AutoModelForCausalLM).
        tokenizer: Tokenizador correspondiente al modelo.
        prompt (str): Texto de entrada para la generación.
        max_length (int): Longitud máxima del texto generado.
        temperature (float): Controla la aleatoriedad (menor valor es más conservador).
        top_p (float): Nucleus sampling, limita la generación a los tokens más probables.
        repetition_penalty (float): Penaliza repeticiones para evitar texto repetitivo.

    Retorna:
        generated_text (str): Texto generado.
    """
    # Configurar el token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenizar el texto de entrada con máscara de atención
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Mover el modelo y los datos a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generar texto
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )

    # Decodificar y retornar el texto generado
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def load_model_with_lora_adapter(model_name: str, lora_adapter_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Carga el modelo preentrenado y aplica el adaptador LoRA.

    Parámetros:
        model_name (str): Nombre del modelo preentrenado (e.g., "EleutherAI/pythia-70m").
        lora_adapter_path (str): Ruta a la carpeta del adaptador LoRA guardado.
        device (str): Dispositivo para cargar el modelo ("cuda" o "cpu").

    Retorna:
        model (PeftModel): Modelo combinado con el adaptador LoRA.
        tokenizer (AutoTokenizer): Tokenizador correspondiente al modelo.
    """
    # Cargar el tokenizador
    print(f"Cargando el tokenizador para {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configurar el token de padding si no está definido
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cargar el modelo preentrenado
    print(f"Cargando el modelo preentrenado {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Cargar el adaptador LoRA
    print(f"Aplicando el adaptador LoRA desde {lora_adapter_path}...")
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    # Mover el modelo al dispositivo especificado
    model.to(device)

    # Verificar los parámetros entrenables
    model.print_trainable_parameters()

    print("El modelo combinado con el adaptador LoRA ha sido cargado exitosamente.")
    return model, tokenizer
