import random


def select_best_action(results: dict[str, str]) -> str:
    """
    Give all the metadata resuls select the action that is the most popular
    """
    frequencies = {
        "ScanNetwork": 0,
        "ScanServices": 0,
        "ExploitService": 0,
        "FindData": 0,
        "ExfiltrateData": 0,
    }

    for act in results:
        actions = act["action"].split("|")
        for a in actions:
            frequencies[a] += 1

    # return max(votes, key=votes.get)
    return sample_best_action(frequencies)


def sample_best_action(data: dict[str, int]) -> str:
    """
    Given a dictionary of actions and frequencies sample distribution
    to select one of the actions
    """
    total = sum(data.values())

    # Calculate probability distribution
    prob_dist = {key: value / total for key, value in data.items()}

    # Sample keys based on the probability distribution
    sampled_key = random.choices(
        list(prob_dist.keys()), weights=list(prob_dist.values()), k=1
    )[0]
    return sampled_key


# What eles do we need here?
