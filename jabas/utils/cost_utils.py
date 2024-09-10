BASE_TFPLOS = 14.13

# AWS P3 instance (4 V100 GPU) price per hour (dollar) = 12.24
# Price of single GPU per hour = 12.24 / 4 = 3.06
BASE_COST = 3.06


def cost_model(tfplos):
    return BASE_COST * (tfplos/BASE_TFPLOS)


def estimate_cost(tfplos, num_gpus, time):
    return cost_model(tfplos) * num_gpus * time