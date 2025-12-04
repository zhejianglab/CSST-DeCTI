import os


def check_dist_env():

    for k in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        if k not in os.environ:
            return False
    if int(os.environ["WORLD_SIZE"]) < 1:
        return False
    if int(os.environ["RANK"]) < 0 or int(os.environ["LOCAL_RANK"]) < 0:
        return False

    return True


def setup_dist_env(master_addr="localhost", master_port="12345"):

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
