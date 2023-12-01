



import pytest
import torch
from torchmdnet.models.model import create_model
from utils import load_example_args, create_example_batch
import random
import numpy as np

def set_all_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed+1)
    np.random.seed(seed+2)

def test_tensorforce():
    args = load_example_args("tensornet", remove_prior=True)
    args["model"] = "tensornet"
    args["manual_grad"] = False
    args["derivative"] = True
    args["output_model"] = "Scalar"
    args["seed"] = 1234
    args["num_layers"] = 0
    set_all_seeds(args["seed"])
    model_truth = create_model(args)

    args["model"] = "tensorforcenet"
    args["output_model"] = "Identity"
    args["manual_grad"] = True
    args["derivative"] = False
    set_all_seeds(args["seed"])
    model_test = create_model(args)

    n_atoms = 200
    z, pos, batch = create_example_batch(n_atoms=n_atoms)

    energy, forces = model_truth(z, pos, batch)
    del model_truth
    energy_test, forces_test = model_test(z, pos, batch)
    del model_test
    enrgy_rel_error = torch.abs(energy - energy_test) / torch.abs(energy)
    forces_rel_error = torch.abs(forces - forces_test) / torch.abs(forces)

    print("Max energy relative error: ", torch.max(enrgy_rel_error))
    print("Max forces relative error: ", torch.max(forces_rel_error))
    print("Mean energy relative error: ", torch.mean(enrgy_rel_error))
    print("Mean forces relative error: ", torch.mean(forces_rel_error))


    torch.testing.assert_allclose(energy, energy_test)
    torch.testing.assert_allclose(forces, forces_test)
