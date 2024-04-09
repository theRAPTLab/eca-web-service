from dataclasses import dataclass, field
import json
from enum import Enum
import dataclasses


@dataclass
class ModelArguments:
    # training parameters
    seed: int = field(default=34, metadata={"help": "random seed for reproducibility of results"})
    run_name: str = field(default="test_", metadata={"help": "the name of run. files will be saved by this name"})
    dryrun: bool = field(default=True, metadata={"help": "if we are saving actual files or not"})
    eval_only: bool = field(default=True, metadata={"help": "if we are running only in eval mode. no dataframe or "
                                                            "estimator is needed. but will need a behavior cloning "
                                                            "model and a q network saved"})

    state_dim: float = field(default=6, metadata={"help": "length of state dimension"})
    action_dim: float = field(default=2, metadata={"help": "number of actions dimension"})
    discount: float = field(default=0.99, metadata={"help": "discounted factor or gamma for the narrative planner"})

    units: float = field(default=128, metadata={"help": "number of neurons per layer"})
    lr: float = field(default=0.001, metadata={"help": "model learning rate"})
    train_steps: int = field(default=int(1e5), metadata={"help": "total number of training steps"})
    bcq_threshold: float = field(default=0.1, metadata={"help": "bcq model constraint parameter (tau value)"})
    update_frequency: int = field(default=int(1e3), metadata={"help": "update target network frequency"})
    batch_size: int = field(default=256, metadata={"help": "batch size for training narrative planner"})
    device: str = field(default="cpu", metadata={"help": "device (cpu|cuda:0)"})

    log_frequency: int = field(default=int(1e3), metadata={"help": "log model loss and other metrics"})

    # aes parameters
    bc_location: str = field(default="../checkpoint/providehint_seed1_bcq_bc.ckpt", metadata={"help": "location of the behavior cloning file"})

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)
