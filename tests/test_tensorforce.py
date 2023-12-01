



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
    args["seed"] = 12345
    args["num_layers"] = 0
    args["precision"] = 32
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

    if args["precision"] == 64:
        pos = pos.to(torch.float64)
    pos.requires_grad = True
    energy, forces = model_truth(z, pos, batch)
    (forces + energy.sum()).sum().backward()
    force_diff = pos.grad.clone().detach()
    pos.grad = None
    del model_truth
    energy_test, forces_test = model_test(z, pos, batch)
    (forces_test+energy_test.sum()).sum().backward()
    force_diff_test = pos.grad.clone().detach()
    del model_test
    enrgy_rel_error = torch.abs(energy - energy_test) / torch.abs(energy)
    forces_rel_error = torch.abs(forces - forces_test) / torch.abs(forces)
    force_diff_rel_error = torch.abs(force_diff - force_diff_test) / torch.abs(force_diff)

    print("Max energy relative error: ", torch.max(enrgy_rel_error))
    print("Max forces relative error in X direction: ", torch.max(forces_rel_error[:, 0]))
    print("Max forces relative error in Y direction: ", torch.max(forces_rel_error[:, 1]))
    print("Max forces relative error in Z direction: ", torch.max(forces_rel_error[:, 2]))
    print("Max force diff relative error: ", torch.max(force_diff_rel_error))
    print("Mean energy relative error: ", torch.mean(enrgy_rel_error))
    print("Mean forces relative error: ", torch.mean(forces_rel_error))
    print("Mean force diff relative error: ", torch.mean(force_diff_rel_error))

    torch.testing.assert_allclose(energy, energy_test)
    torch.testing.assert_allclose(forces, forces_test)
    torch.testing.assert_allclose(force_diff, force_diff_test)
