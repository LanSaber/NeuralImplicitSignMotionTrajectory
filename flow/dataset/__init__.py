__all__ = ["UpperSMPLXFlowDataset", "collate_upper_smplx", "read_jsonl"]


def __getattr__(name):
    if name in __all__:
        from flow.dataset import upper_smplx

        return getattr(upper_smplx, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
