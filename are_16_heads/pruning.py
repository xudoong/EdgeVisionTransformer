from math import sqrt
from logger import logger


def parse_head_pruning_descriptors(
    descriptors,
    reverse_descriptors=False,
    n_heads=None
):
    """Returns a dictionary mapping layers to the set of heads to prune in
    this layer"""
    to_prune = {}
    for descriptor in descriptors:
        layer, heads = descriptor.split(":")
        layer = int(layer) - 1
        heads = set(int(head) - 1 for head in heads.split(","))
        if layer not in to_prune:
            to_prune[layer] = set()
        to_prune[layer].update(heads)
    # Reverse
    if reverse_descriptors:
        if n_heads is None:
            raise ValueError("You need to specify the total number of heads")
        for layer, heads in to_prune.items():
            to_prune[layer] = set([head for head in range(n_heads)
                                   if head not in heads])
    return to_prune


def to_pruning_descriptor(to_prune):
    """Inverse of parse_head_pruning_descriptors"""
    descriptors = [f"{layer+1}:{','.join(str(head+1) for head in heads)}"
                   for layer, heads in to_prune.items()]
    return " ".join(descriptors)


def determine_pruning_sequence(
    prune_numbers,
    prune_percents,
    n_heads,
    n_layers,
    at_least_x_heads_per_layer=0,
):
    all_n_to_prune = prune_numbers
    if all_n_to_prune is None:
        # Compute the number of heads to prune on percentage if needed
        all_n_to_prune = []
        for prune_percent in prune_percents:
            total_heads = n_heads * n_layers
            n_to_prune = int(total_heads * prune_percent / 100)
            # Make sure we keep at least one head per layer
            if at_least_x_heads_per_layer > 0:
                if n_to_prune > total_heads - at_least_x_heads_per_layer * n_layers:
                    logger.warn(
                        f"Can't prune {prune_percent}% ({n_to_prune})"
                        f" heads AND keep at least {at_least_x_heads_per_layer}"
                        " head(s) per layer. Will"
                        f" prune only {(1-(at_least_x_heads_per_layer*n_layers)/total_heads)*100:.1f} "
                        f"({total_heads-n_layers}) heads instead"
                    )
                    n_to_prune = total_heads - at_least_x_heads_per_layer * n_layers
                    all_n_to_prune.append(n_to_prune)
                    break
            all_n_to_prune.append(n_to_prune)

    # We'll incrementally prune layers and evaluate
    all_n_to_prune = sorted(all_n_to_prune)
    n_to_prune_sequence = all_n_to_prune[:]
    for idx in range(1, len(all_n_to_prune)):
        n_to_prune_sequence[idx] = all_n_to_prune[idx] - all_n_to_prune[idx-1]
    # Verify that the total number of heads pruned stayed the same
    assert all_n_to_prune[-1] == sum(n_to_prune_sequence)
    return n_to_prune_sequence


def what_to_prune(
    head_importance,
    n_to_prune,
    to_prune=None,
    at_least_x_heads_per_layer=0,
    rescale_by_number=False,
):
    head_importance = head_importance.clone()
    n_layers, n_heads = head_importance.size()
    to_prune = to_prune or {}
    if rescale_by_number:
        for layer in to_prune:
            #head_importance[layer] *= sqrt(n_layers / len(to_prune[layer]))
            head_importance[layer] *= sqrt(len(to_prune[layer]) / n_layers)
    # Sort heads by score
    heads_and_score = [
        ((layer, head), head_importance[layer, head])
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    heads_and_score = sorted(heads_and_score, key=lambda x: x[1])
    sorted_heads = [head_and_score[0]
                    for head_and_score in heads_and_score]
    # Ensure we don't delete all heads in a layer
    if at_least_x_heads_per_layer:
        # Remove the top scoring head in each layer
        to_protect = {l: 0 for l in range(n_layers)}
        filtered_sorted_heads = []
        for layer, head in reversed(sorted_heads):
            if layer in to_protect:
                if to_protect[layer] < at_least_x_heads_per_layer:
                    to_protect[layer] += 1
                    continue
                else:
                    to_protect.pop(layer)
            filtered_sorted_heads.insert(0, (layer, head))
        sorted_heads = filtered_sorted_heads
    # layer/heads that were already pruned
    # Prune the lowest scoring heads
    sorted_heads = [
        (layer, head)
        for (layer, head) in sorted_heads
        if layer not in to_prune or head not in to_prune[layer]
    ]
    # Update heads to prune
    for layer, head in sorted_heads[:n_to_prune]:
        if layer not in to_prune:
            to_prune[layer] = set()
        to_prune[layer].add(head)
    return to_prune
