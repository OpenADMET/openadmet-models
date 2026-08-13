"""Batch size utilities."""

def safe_inference_batch_size(dataset_size: int, batch_size: int) -> int:
    """
    Return a batch size that avoids chemprop's silent single-molecule drop.

    Chemprop's build_dataloader drops the last batch when dataset_size % batch_size == 1
    to protect batch-norm during training. At inference that would silently omit a molecule
    and misalign embeddings with input order, so shrink the batch size until the remainder
    condition no longer holds.

    Parameters
    ----------
    dataset_size : int
        Number of molecules to batch.
    batch_size : int
        Requested batch size.

    Returns
    -------
    int
        A batch size no larger than dataset_size for which dataset_size % batch_size != 1
        (or 1 if no larger value works).

    """
    effective = min(batch_size, dataset_size)
    while effective > 1 and dataset_size % effective == 1:
        effective -= 1
    return effective
