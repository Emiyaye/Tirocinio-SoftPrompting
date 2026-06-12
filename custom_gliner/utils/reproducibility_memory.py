import os
import random

import numpy as np
import torch

SEED = 42

class ReproducibilityManager:

    def __init__(self, seed=SEED, deterministic_algorithms=True, benchmark_mode=False):
        self.seed = seed
        self.deterministic_algorithms = deterministic_algorithms
        self.benchmark_mode = benchmark_mode

    def set_seed(self, seed=None, verbose=True):
        if seed is not None:
            self.seed = seed

        os.environ["PYTHONHASHSEED"] = str(self.seed)
        if self.deterministic_algorithms:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Set Python random seed
        random.seed(self.seed)

        # Set NumPy random seed
        np.random.seed(self.seed)

        # Set PyTorch random seeds
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups

        # Configure PyTorch for reproducibility
        if self.deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Configure cuDNN
        torch.backends.cudnn.deterministic = self.deterministic_algorithms and not self.benchmark_mode
        torch.backends.cudnn.benchmark = self.benchmark_mode and not self.deterministic_algorithms

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

        if verbose:
            print(f"- Reproducibility seed set to: {self.seed}")
            print(f"- Deterministic algorithms: {self.deterministic_algorithms}")
            print(f"- cuDNN benchmark mode: {self.benchmark_mode}")

    def print_environment_info(self):
        """
        Print information about the current environment and reproducibility settings.
        """
        print("=" * 60)
        print("REPRODUCIBILITY ENVIRONMENT INFO")
        print("=" * 60)
        print(f"Seed: {self.seed}")
        print(f"Deterministic algorithms: {self.deterministic_algorithms}")
        print(f"cuDNN benchmark: {self.benchmark_mode}")
        print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
        print(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set')}")
        print("=" * 60)
