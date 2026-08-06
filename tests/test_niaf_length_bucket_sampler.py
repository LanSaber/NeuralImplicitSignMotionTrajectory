from collections import Counter

from NIAF.continuous_sign_field.data import LengthBucketDistributedSampler


def _rank_batches(sampler, batch_size):
    values = list(iter(sampler))
    return [values[offset : offset + batch_size] for offset in range(0, len(values), batch_size)]


def test_length_bucket_sampler_balances_ddp_ranks_and_covers_dataset():
    lengths = [40 + index // 8 for index in range(67)] + [204, 260, 378]
    batch_size = 4
    samplers = [
        LengthBucketDistributedSampler(
            lengths,
            batch_size=batch_size,
            num_replicas=4,
            rank=rank,
            shuffle=True,
            seed=17,
            pad_to_full_batch=True,
        )
        for rank in range(4)
    ]
    rank_batches = [_rank_batches(sampler, batch_size) for sampler in samplers]

    assert len({len(sampler) for sampler in samplers}) == 1
    assert len({len(batches) for batches in rank_batches}) == 1

    observed = Counter()
    for batch_index in range(len(rank_batches[0])):
        global_batch = []
        rank_maxima = []
        for batches in rank_batches:
            local_batch = batches[batch_index]
            global_batch.extend(local_batch)
            rank_maxima.append(max(lengths[index] for index in local_batch))
        observed.update(global_batch)
        assert max(rank_maxima) - min(rank_maxima) <= 1

    assert set(observed) == set(range(len(lengths)))
    assert sum(observed.values()) - len(lengths) < batch_size * 4


def test_length_bucket_sampler_changes_batch_order_by_epoch_deterministically():
    lengths = [40 + index // 4 for index in range(128)]
    sampler = LengthBucketDistributedSampler(
        lengths,
        batch_size=8,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=31,
    )
    epoch_zero = list(iter(sampler))
    assert epoch_zero == list(iter(sampler))

    sampler.set_epoch(1)
    epoch_one = list(iter(sampler))
    assert epoch_one != epoch_zero
