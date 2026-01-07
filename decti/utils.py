import os
import random

__all__ = ["check_dist_env", "setup_dist_env", "shuffle_paired_lists"]


def check_dist_env():

    for k in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        if k not in os.environ:
            return False
    if int(os.environ["WORLD_SIZE"]) < 1:
        return False
    if int(os.environ["RANK"]) < 0 or int(os.environ["LOCAL_RANK"]) < 0:
        return False

    return True


def setup_dist_env(
    master_addr: str = "localhost",
    master_port: str = "12345",
):

    if not check_dist_env():
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
    if int(os.environ["WORLD_SIZE"]) < 1:
        raise Exception("Invalid init_process_group parameter")

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = max(int(os.environ.get("LOCAL_RANK", -1)), 0)

    return world_size, rank, local_rank


def shuffle_paired_lists(input_paths, target_paths, validate_ratio, seed):

    # shuffle input image lists
    combined = list(zip(input_paths, target_paths))
    random.seed(seed)
    random.shuffle(combined)
    input_paths, target_paths = zip(*combined)
    n = len(input_paths)

    n_validate = max(int(n * validate_ratio + 0.5), 1)
    n_train = n - n_validate
    if n_train == 0:
        raise Exception("too few input images")

    input_tr = input_paths[0:n_train]
    target_tr = target_paths[0:n_train]
    input_va = input_paths[n_train:]
    target_va = target_paths[n_train:]

    return input_tr, target_tr, input_va, target_va
