"""Fail-closed network evidence for the paired two-node Stage-C pilot.

The node audit is intentionally stdlib-only.  Torch is imported only by the
NCCL smoke subcommand after both physical RoCE rails have been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import socket
import statistics
import subprocess
import time
from pathlib import Path


SCHEMA = "signtrajfield_stage_c_paired_roce_preflight"
SCHEMA_VERSION = 1
GID_INDEX = 5
MINIMUM_SPEED_MBIT = 200_000
MINIMUM_MTU = 9_000
RAILS = (
    ("rocep1s0f1", "enp1s0f1np1"),
    ("roceP2p1s0f1", "enP2p1s0f1np1"),
)
PAIR_RE = re.compile(r"^pair(0[1-9]|1[0-5])$")
NCCL_PROFILES = {
    "single_primary": "rocep1s0f1:1",
    "single_secondary": "roceP2p1s0f1:1",
    "dual": "rocep1s0f1:1,roceP2p1s0f1:1",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The first payload is the exact Stage-C trainable gradient size audited from
# the seven approved generator prefixes.  The larger payloads expose whether
# dual-rail bandwidth helps beyond tiny collective latency.
PAYLOADS = (
    ("stage_c_gradient", 1_109_395, 3, 7),
    ("64_mib", 64 * 1024 * 1024 // 4, 2, 5),
    ("256_mib", 256 * 1024 * 1024 // 4, 1, 3),
)
NETWORK_COUNTERS = (
    "rx_bytes",
    "tx_bytes",
    "rx_packets",
    "tx_packets",
    "rx_errors",
    "tx_errors",
    "rx_dropped",
    "tx_dropped",
    "rx_crc_errors",
    "rx_fifo_errors",
    "tx_fifo_errors",
    "rx_missed_errors",
    "rx_over_errors",
    "tx_aborted_errors",
    "tx_carrier_errors",
)
HCA_COUNTERS = (
    "counters/port_rcv_data",
    "counters/port_xmit_data",
    "counters/port_rcv_packets",
    "counters/port_xmit_packets",
    "counters/port_rcv_errors",
    "counters/port_xmit_discards",
    "counters/symbol_error",
    "counters/link_downed",
    "counters/link_error_recovery",
    "counters/local_link_integrity_errors",
    "counters/excessive_buffer_overrun_errors",
    "hw_counters/out_of_buffer",
    "hw_counters/rx_icrc_encapsulated",
    "hw_counters/packet_seq_err",
    "hw_counters/req_cqe_error",
    "hw_counters/resp_cqe_error",
    "hw_counters/roce_adp_retrans_to",
)
ETHTOOL_HARMFUL_COUNTERS = (
    "rx_out_of_buffer",
    "rx_crc_errors_phy",
    "rx_symbol_err_phy",
    "rx_discards_phy",
    "tx_discards_phy",
    "tx_errors_phy",
    "rx_pcs_symbol_err_phy",
    "rx_buffer_passed_thres_phy",
    "tx_pause_storm_warning_events",
    "tx_pause_storm_error_events",
)
HARMFUL_NETWORK_COUNTERS = frozenset(NETWORK_COUNTERS) - {
    "rx_bytes",
    "tx_bytes",
    "rx_packets",
    "tx_packets",
}
HARMFUL_HCA_COUNTERS = frozenset(HCA_COUNTERS) - {
    "counters/port_rcv_data",
    "counters/port_xmit_data",
    "counters/port_rcv_packets",
    "counters/port_xmit_packets",
}


class PreflightError(RuntimeError):
    """The allocation is not the exact paired 200-Gb/s RoCE topology."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise PreflightError(f"cannot read required runtime attribute {path}: {error}") from error


def _require_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PreflightError(f"command failed: {command[0]}: {error}") from error
    return result.stdout


def _atomic_create_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _ipv4_interface(netdev: str) -> ipaddress.IPv4Interface:
    output = _require_command(["ip", "-j", "-4", "address", "show", "dev", netdev])
    try:
        records = json.loads(output)
        addresses = [
            ipaddress.IPv4Interface(f"{row['local']}/{row['prefixlen']}")
            for record in records
            for row in record.get("addr_info", ())
            if row.get("family") == "inet" and row.get("scope") == "global"
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid IPv4 metadata for {netdev}") from error
    if len(addresses) != 1 or addresses[0].network.prefixlen != 30:
        raise PreflightError(
            f"{netdev} must have exactly one global /30 IPv4 address; got {addresses}"
        )
    return addresses[0]


def audit_node(output_dir: Path) -> dict:
    host = socket.gethostname().split(".", 1)[0]
    rails: dict[str, dict] = {}
    for hca, netdev in RAILS:
        port = Path("/sys/class/infiniband") / hca / "ports/1"
        network = Path("/sys/class/net") / netdev
        state = _read(port / "state")
        physical_state = _read(port / "phys_state")
        link_layer = _read(port / "link_layer")
        rate = _read(port / "rate")
        operstate = _read(network / "operstate")
        speed = int(_read(network / "speed"))
        mtu = int(_read(network / "mtu"))
        gid_type = _read(port / f"gid_attrs/types/{GID_INDEX}")
        gid_netdev = _read(port / f"gid_attrs/ndevs/{GID_INDEX}")
        gid = ipaddress.IPv6Address(_read(port / f"gids/{GID_INDEX}"))
        ipv4 = _ipv4_interface(netdev)
        ibv = _require_command(["ibv_devinfo", "-d", hca])

        if not state.endswith("ACTIVE") or not physical_state.endswith("LinkUp"):
            raise PreflightError(f"{hca}:1 is not ACTIVE/LinkUp: {state!r} {physical_state!r}")
        if (
            link_layer != "Ethernet"
            or "PORT_ACTIVE" not in ibv
            or re.search(r"link_layer:\s+Ethernet", ibv) is None
        ):
            raise PreflightError(f"{hca}:1 is not an active RoCE Ethernet verbs port")
        rate_match = re.match(r"^(\d+(?:\.\d+)?) Gb/sec", rate)
        if rate_match is None or float(rate_match.group(1)) < 200.0:
            raise PreflightError(f"{hca}:1 rate is below 200 Gb/sec: {rate!r}")
        if operstate != "up" or speed < MINIMUM_SPEED_MBIT or mtu < MINIMUM_MTU:
            raise PreflightError(
                f"{netdev} is not up at >=200000 Mbit/s with >=9000 MTU: "
                f"state={operstate!r} speed={speed} mtu={mtu}"
            )
        if gid_type != "RoCE v2" or gid_netdev != netdev:
            raise PreflightError(
                f"{hca} GID {GID_INDEX} is not RoCE-v2 on {netdev}: "
                f"type={gid_type!r} ndev={gid_netdev!r}"
            )
        if gid.ipv4_mapped != ipv4.ip:
            raise PreflightError(
                f"{hca} GID {GID_INDEX} does not encode {netdev} IPv4 {ipv4.ip}: {gid}"
            )
        rails[hca] = {
            "port": 1,
            "state": state,
            "physical_state": physical_state,
            "link_layer": link_layer,
            "rate": rate,
            "netdev": netdev,
            "operstate": operstate,
            "speed_mbit": speed,
            "mtu": mtu,
            "ipv4_interface": str(ipv4),
            "gid_index": GID_INDEX,
            "gid_type": gid_type,
            "gid": str(gid),
        }

    payload = {
        "schema_name": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "host": host,
        "rails": rails,
    }
    _atomic_create_json(output_dir / f"{host}.node.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def _load_node_manifests(directory: Path, nodes: list[str]) -> dict[str, dict]:
    if any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", node) is None
        for node in nodes
    ):
        raise PreflightError("node-audit host names are malformed")
    expected = {f"{node}.node.json" for node in nodes}
    observed = {path.name for path in directory.glob("*.node.json")}
    if observed != expected:
        raise PreflightError(
            f"node-audit evidence mismatch: expected={sorted(expected)} observed={sorted(observed)}"
        )
    values = {}
    for node in nodes:
        value = json.loads((directory / f"{node}.node.json").read_text(encoding="utf-8"))
        if value.get("schema_name") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
            raise PreflightError(f"invalid node-audit schema for {node}")
        if value.get("host") != node or set(value.get("rails", {})) != {
            item[0] for item in RAILS
        }:
            raise PreflightError(f"node-audit identity mismatch for {node}")
        for hca, netdev in RAILS:
            rail = value["rails"][hca]
            expected_fields = {
                "port",
                "state",
                "physical_state",
                "link_layer",
                "rate",
                "netdev",
                "operstate",
                "speed_mbit",
                "mtu",
                "ipv4_interface",
                "gid_index",
                "gid_type",
                "gid",
            }
            try:
                rate_match = re.fullmatch(
                    r"(\d+(?:\.\d+)?) Gb/sec.*", str(rail.get("rate", ""))
                )
                interface = ipaddress.IPv4Interface(
                    str(rail.get("ipv4_interface", ""))
                )
                gid = ipaddress.IPv6Address(str(rail.get("gid", "")))
            except (ipaddress.AddressValueError, ValueError):
                raise PreflightError(
                    f"node-audit address/rate evidence is invalid for {node}/{hca}"
                ) from None
            if (
                set(rail) != expected_fields
                or rail.get("port") != 1
                or not str(rail.get("state", "")).endswith("ACTIVE")
                or not str(rail.get("physical_state", "")).endswith("LinkUp")
                or rail.get("link_layer") != "Ethernet"
                or rate_match is None
                or float(rate_match.group(1)) < 200.0
                or rail.get("netdev") != netdev
                or rail.get("operstate") != "up"
                or isinstance(rail.get("speed_mbit"), bool)
                or not isinstance(rail.get("speed_mbit"), int)
                or rail["speed_mbit"] < MINIMUM_SPEED_MBIT
                or isinstance(rail.get("mtu"), bool)
                or not isinstance(rail.get("mtu"), int)
                or rail["mtu"] < MINIMUM_MTU
                or interface.network.prefixlen != 30
                or rail.get("gid_index") != GID_INDEX
                or rail.get("gid_type") != "RoCE v2"
                or gid.ipv4_mapped != interface.ip
            ):
                raise PreflightError(
                    f"node-audit rail health is invalid for {node}/{hca}"
                )
        values[node] = value
    return values


def validate_pair_evidence(directory: Path, nodes: list[str], pair: str) -> dict:
    """Recompute exact pair evidence from the two raw node audits."""

    match = PAIR_RE.fullmatch(pair)
    if match is None or len(nodes) != 2 or len(set(nodes)) != 2:
        raise PreflightError("pair validation requires one pair01..pair15 and two distinct nodes")
    values = _load_node_manifests(directory, nodes)
    pair_number = int(match.group(1))
    expected_networks = (
        ipaddress.IPv4Network(f"10.250.{pair_number}.0/30"),
        ipaddress.IPv4Network(f"10.250.{100 + pair_number}.0/30"),
    )
    rail_evidence = []
    for (hca, netdev), expected_network in zip(RAILS, expected_networks, strict=True):
        interfaces = [
            ipaddress.IPv4Interface(values[node]["rails"][hca]["ipv4_interface"])
            for node in nodes
        ]
        if any(interface.network != expected_network for interface in interfaces):
            raise PreflightError(
                f"{hca} does not join both nodes on expected {pair} network {expected_network}"
            )
        if interfaces[0].ip == interfaces[1].ip:
            raise PreflightError(f"{hca} has duplicate endpoint IPs")
        rail_evidence.append(
            {
                "hca": hca,
                "netdev": netdev,
                "network": str(expected_network),
                "endpoints": {node: str(interface.ip) for node, interface in zip(nodes, interfaces, strict=True)},
            }
        )

    payload = {
        "schema_name": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "pair_constraint": pair,
        "nodes": nodes,
        "rails": rail_evidence,
        "minimum_speed_mbit": MINIMUM_SPEED_MBIT,
        "gid_index": GID_INDEX,
    }
    return payload


def validate_pair(directory: Path, nodes: list[str], pair: str) -> dict:
    payload = validate_pair_evidence(directory, nodes, pair)
    _atomic_create_json(directory / "PAIR.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def verify_peer(directory: Path) -> dict:
    pair = json.loads((directory / "PAIR.json").read_text(encoding="utf-8"))
    host = socket.gethostname().split(".", 1)[0]
    nodes = pair.get("nodes")
    if not isinstance(nodes, list) or host not in nodes or len(nodes) != 2:
        raise PreflightError(f"host {host} is not one endpoint of the exact pair")
    peer = next(node for node in nodes if node != host)
    checks = []
    for rail in pair.get("rails", ()):
        netdev = rail["netdev"]
        peer_ip = rail["endpoints"][peer]
        _require_command(["ping", "-n", "-I", netdev, "-c", "2", "-W", "2", peer_ip])
        checks.append({"netdev": netdev, "peer": peer, "peer_ipv4": peer_ip, "passed": True})
    payload = {
        "schema_name": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "host": host,
        "pair_constraint": pair["pair_constraint"],
        "peer_connectivity": checks,
    }
    _atomic_create_json(directory / f"{host}.connectivity.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def validate_peer_evidence(directory: Path, pair: dict) -> dict[str, dict]:
    """Reopen the exact two directional peer-connectivity attestations."""

    nodes = pair.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 2 or len(set(nodes)) != 2:
        raise PreflightError("peer evidence requires exactly two pair nodes")
    expected_names = {f"{node}.connectivity.json" for node in nodes}
    observed_names = {path.name for path in directory.glob("*.connectivity.json")}
    if observed_names != expected_names:
        raise PreflightError(
            "peer-connectivity evidence set is not exact: "
            f"expected={sorted(expected_names)} observed={sorted(observed_names)}"
        )
    result = {}
    for host in nodes:
        value = json.loads(
            (directory / f"{host}.connectivity.json").read_text(encoding="utf-8")
        )
        peer = next(node for node in nodes if node != host)
        expected_checks = [
            {
                "netdev": rail["netdev"],
                "peer": peer,
                "peer_ipv4": rail["endpoints"][peer],
                "passed": True,
            }
            for rail in pair["rails"]
        ]
        if value != {
            "schema_name": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "host": host,
            "pair_constraint": pair["pair_constraint"],
            "peer_connectivity": expected_checks,
        }:
            raise PreflightError(f"peer-connectivity evidence changed for {host}")
        result[host] = value
    return result


def _integer_counter(path: Path) -> int:
    value = _read(path)
    try:
        parsed = int(value)
    except ValueError as error:
        raise PreflightError(f"counter is not an integer: {path}={value!r}") from error
    if parsed < 0:
        raise PreflightError(f"counter is negative: {path}={parsed}")
    return parsed


def _ethtool_counters(netdev: str) -> dict[str, int]:
    output = _require_command(["ethtool", "-S", netdev])
    observed: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"^\s*([^:]+):\s*([0-9]+)\s*$", line)
        if match and match.group(1) in ETHTOOL_HARMFUL_COUNTERS:
            observed[match.group(1)] = int(match.group(2))
    missing = sorted(set(ETHTOOL_HARMFUL_COUNTERS) - set(observed))
    if missing:
        raise PreflightError(
            f"{netdev} does not expose required ConnectX health counters: {missing}"
        )
    return observed


def snapshot_counters(directory: Path, phase: str) -> dict:
    if phase not in {"pre", "post"}:
        raise PreflightError("counter snapshot phase must be pre or post")
    host = socket.gethostname().split(".", 1)[0]
    rails = {}
    for hca, netdev in RAILS:
        network_root = Path("/sys/class/net") / netdev / "statistics"
        hca_root = Path("/sys/class/infiniband") / hca / "ports/1"
        rails[hca] = {
            "netdev": netdev,
            "network": {
                name: _integer_counter(network_root / name)
                for name in NETWORK_COUNTERS
            },
            "verbs": {
                name: _integer_counter(hca_root / name) for name in HCA_COUNTERS
            },
            "ethtool": _ethtool_counters(netdev),
        }
    payload = {
        "schema_name": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "host": host,
        "phase": phase,
        "rails": rails,
    }
    _atomic_create_json(directory / f"{host}.counters_{phase}.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def validate_counter_snapshots(directory: Path, nodes: list[str]) -> dict:
    """Recompute exact counter health from all four raw snapshots."""

    if len(nodes) != 2 or len(set(nodes)) != 2:
        raise PreflightError("counter comparison requires exactly two distinct nodes")
    harmful = []
    traffic = []
    for node in nodes:
        snapshots = {}
        for phase in ("pre", "post"):
            path = directory / f"{node}.counters_{phase}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("schema_name") != SCHEMA
                or value.get("schema_version") != SCHEMA_VERSION
                or value.get("host") != node
                or value.get("phase") != phase
            ):
                raise PreflightError(f"invalid {phase} counter snapshot for {node}")
            snapshots[phase] = value
        for hca, netdev in RAILS:
            before = snapshots["pre"]["rails"][hca]
            after = snapshots["post"]["rails"][hca]
            if before["netdev"] != netdev or after["netdev"] != netdev:
                raise PreflightError(f"counter rail identity changed for {node}/{hca}")
            for family, names in (
                ("network", NETWORK_COUNTERS),
                ("verbs", HCA_COUNTERS),
                ("ethtool", ETHTOOL_HARMFUL_COUNTERS),
            ):
                if set(before[family]) != set(names) or set(after[family]) != set(names):
                    raise PreflightError(f"counter set changed for {node}/{hca}/{family}")
                for name in names:
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in (before[family][name], after[family][name])
                    ):
                        raise PreflightError(
                            f"counter value is invalid: {node}/{hca}/{family}/{name}"
                        )
                    delta = after[family][name] - before[family][name]
                    if delta < 0:
                        raise PreflightError(
                            f"counter reset prevents health proof: {node}/{hca}/{family}/{name}"
                        )
                    row = {
                        "node": node,
                        "hca": hca,
                        "netdev": netdev,
                        "family": family,
                        "counter": name,
                        "before": before[family][name],
                        "after": after[family][name],
                        "delta": delta,
                    }
                    is_harmful = (
                        (family == "network" and name in HARMFUL_NETWORK_COUNTERS)
                        or (family == "verbs" and name in HARMFUL_HCA_COUNTERS)
                        or family == "ethtool"
                    )
                    (harmful if is_harmful else traffic).append(row)
    increments = [row for row in harmful if row["delta"] != 0]
    if increments:
        raise PreflightError(f"harmful RoCE counter increments detected: {increments}")
    payload = {
        "schema_name": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "nodes": nodes,
        "harmful_counter_increment_count": 0,
        "harmful_counters": harmful,
        "traffic_counters": traffic,
    }
    return payload


def compare_counters(directory: Path, nodes: list[str]) -> dict:
    payload = validate_counter_snapshots(directory, nodes)
    _atomic_create_json(directory / "COUNTER_HEALTH.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def verify_nccl_logs(directory: Path, prefix: str, profile: str, output_file: Path) -> dict:
    if profile not in NCCL_PROFILES or not prefix or Path(prefix).name != prefix:
        raise PreflightError("invalid NCCL-log verification request")
    logs = sorted(directory.glob(f"{prefix}*.log"))
    if len(logs) != 2 or any(not path.is_file() or path.is_symlink() for path in logs):
        raise PreflightError(
            f"expected exactly two regular NCCL logs for {prefix!r}; got {logs}"
        )
    expected_hcas = [item.split(":", 1)[0] for item in NCCL_PROFILES[profile].split(",")]
    per_log_evidence = []
    for path in logs:
        suffix = path.name.removeprefix(prefix).removesuffix(".log")
        host, separator, process_id = suffix.rpartition(".")
        if (
            not separator
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host) is None
            or re.fullmatch(r"[0-9]+", process_id) is None
        ):
            raise PreflightError(
                f"NCCL rank log name lacks exact host/process identity: {path}"
            )
        contents = path.read_text(encoding="utf-8", errors="replace")
        ib_lines = [line for line in contents.splitlines() if "NET/IB" in line]
        if not ib_lines:
            raise PreflightError(
                f"NCCL rank log {path} for {profile} has no positive NET/IB evidence"
            )
        if "NET/Socket" in contents or "Using network Socket" in contents:
            raise PreflightError(
                f"NCCL socket fallback is present in {profile} rank log {path}"
            )
        missing = [
            hca for hca in expected_hcas if not any(hca in line for line in ib_lines)
        ]
        if missing:
            raise PreflightError(
                f"NCCL NET/IB rank log {path} omits intended {profile} HCA(s): "
                f"{missing}"
            )
        per_log_evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "host": host,
                "net_ib_line_count": len(ib_lines),
                "intended_hcas_present": expected_hcas,
                "socket_fallback_detected": False,
            }
        )
    hosts = sorted(row["host"] for row in per_log_evidence)
    if len(set(hosts)) != 2:
        raise PreflightError(f"NCCL logs do not represent two distinct hosts: {hosts}")
    payload = {
        "schema_name": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "intended_hcas": expected_hcas,
        "logs": [str(path.resolve()) for path in logs],
        "hosts": hosts,
        "net_ib_line_count": sum(
            row["net_ib_line_count"] for row in per_log_evidence
        ),
        "socket_fallback_detected": False,
        "per_log_evidence": per_log_evidence,
    }
    _atomic_create_json(output_file, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def distributed_smoke(output_file: Path, profile: str) -> dict | None:
    if profile not in NCCL_PROFILES:
        raise PreflightError(f"unknown NCCL rail profile: {profile}")
    required_environment = {
        "NCCL_NET": "IB",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": NCCL_PROFILES[profile],
        "NCCL_IB_GID_INDEX": str(GID_INDEX),
        "NCCL_SOCKET_IFNAME": "enp1s0f1np1,enP2p1s0f1np1",
        "GLOO_SOCKET_IFNAME": "enp1s0f1np1",
    }
    mismatches = {
        name: (os.environ.get(name), expected)
        for name, expected in required_environment.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise PreflightError(f"fail-closed NCCL environment mismatch: {mismatches}")

    import torch
    import torch.distributed as dist

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    if world_size != 2:
        raise PreflightError(f"NCCL smoke requires world_size=2, got {world_size}")
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=os.environ["SLURM_LOCALID"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PreflightError("each Stage-C rank must see exactly one CUDA device")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        benchmark = []
        for label, elements, warmup_iterations, timed_iterations in PAYLOADS:
            value = torch.empty(elements, dtype=torch.float32, device="cuda")
            for _ in range(warmup_iterations):
                value.fill_(float(rank + 1))
                dist.barrier()
                dist.all_reduce(value)
                torch.cuda.synchronize()
                if not bool(torch.all(value == 3.0).item()):
                    raise PreflightError(
                        f"{profile}/{label} warmup all-reduce failed exact sum check"
                    )

            maximum_times = []
            for _ in range(timed_iterations):
                value.fill_(float(rank + 1))
                dist.barrier()
                torch.cuda.synchronize()
                started = time.perf_counter()
                dist.all_reduce(value)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if not bool(torch.all(value == 3.0).item()):
                    raise PreflightError(
                        f"{profile}/{label} timed all-reduce failed exact sum check"
                    )
                maximum = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                maximum_times.append(float(maximum.item()))
            payload_bytes = elements * 4
            median_seconds = statistics.median(maximum_times)
            benchmark.append(
                {
                    "label": label,
                    "elements": elements,
                    "dtype": "float32",
                    "logical_payload_bytes": payload_bytes,
                    "logical_payload_mib": payload_bytes / (1024**2),
                    "warmup_iterations": warmup_iterations,
                    "timed_iterations": timed_iterations,
                    "maximum_rank_seconds": maximum_times,
                    "median_maximum_rank_seconds": median_seconds,
                    "algorithmic_bandwidth_gbit_s": payload_bytes * 8 / median_seconds / 1e9,
                    "exact_sum_expected": 3.0,
                    "exact_sum_passed": True,
                }
            )
            del value
        hosts: list[str | None] = [None, None]
        dist.all_gather_object(hosts, socket.gethostname().split(".", 1)[0])
        payload = None
        if rank == 0:
            if len(set(hosts)) != 2:
                raise PreflightError(f"NCCL ranks are not on two distinct nodes: {hosts}")
            payload = {
                "schema_name": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "backend": dist.get_backend(),
                "profile": profile,
                "world_size": world_size,
                "hosts": hosts,
                "benchmarks": benchmark,
                "environment": required_environment,
                "ethernet_fallback_permitted": False,
            }
            _atomic_create_json(output_file, payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
        return payload
    finally:
        dist.destroy_process_group()


def validate_benchmarks(
    directory: Path, expected_hosts: list[str] | None = None
) -> dict:
    """Recompute the exact profile decision without publishing a new file."""
    values = {}
    for profile in NCCL_PROFILES:
        path = directory / f"NCCL_BENCHMARK_{profile}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        expected_environment = {
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": NCCL_PROFILES[profile],
            "NCCL_IB_GID_INDEX": str(GID_INDEX),
            "NCCL_SOCKET_IFNAME": "enp1s0f1np1,enP2p1s0f1np1",
            "GLOO_SOCKET_IFNAME": "enp1s0f1np1",
        }
        if (
            value.get("schema_name") != SCHEMA
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get("profile") != profile
            or value.get("backend") != "nccl"
            or value.get("world_size") != 2
            or not isinstance(value.get("hosts"), list)
            or len(value["hosts"]) != 2
            or len(set(value["hosts"])) != 2
            or (
                expected_hosts is not None
                and value["hosts"] != expected_hosts
            )
            or value.get("ethernet_fallback_permitted") is not False
            or value.get("environment") != expected_environment
        ):
            raise PreflightError(f"invalid forced-IB benchmark evidence for {profile}")
        rows = value.get("benchmarks")
        if not isinstance(rows, list) or len(rows) != len(PAYLOADS):
            raise PreflightError(f"incomplete payload benchmark for {profile}")
        for observed, expected in zip(rows, PAYLOADS, strict=True):
            label, elements, warmups, timed = expected
            if (
                observed.get("label") != label
                or observed.get("elements") != elements
                or observed.get("logical_payload_bytes") != elements * 4
                or observed.get("warmup_iterations") != warmups
                or observed.get("timed_iterations") != timed
                or observed.get("exact_sum_passed") is not True
                or observed.get("exact_sum_expected") != 3.0
            ):
                raise PreflightError(f"payload contract mismatch for {profile}/{label}")
            seconds = observed.get("median_maximum_rank_seconds")
            bandwidth = observed.get("algorithmic_bandwidth_gbit_s")
            if not isinstance(seconds, (int, float)) or seconds <= 0:
                raise PreflightError(f"invalid timing for {profile}/{label}")
            if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
                raise PreflightError(f"invalid bandwidth for {profile}/{label}")
            samples = observed.get("maximum_rank_seconds")
            if (
                not isinstance(samples, list)
                or len(samples) != timed
                or any(
                    isinstance(sample, bool)
                    or not isinstance(sample, (int, float))
                    or not (0.0 < float(sample) < float("inf"))
                    for sample in samples
                )
            ):
                raise PreflightError(f"invalid timed samples for {profile}/{label}")
            if label in {"64_mib", "256_mib"} and max(samples) / min(samples) > 2.0:
                raise PreflightError(
                    f"unstable large-payload timings for {profile}/{label}: {samples}"
                )
        values[profile] = value

    comparisons = []
    for index, (label, elements, _, _) in enumerate(PAYLOADS):
        times = {
            profile: values[profile]["benchmarks"][index]["median_maximum_rank_seconds"]
            for profile in NCCL_PROFILES
        }
        fastest_single = min(times["single_primary"], times["single_secondary"])
        comparisons.append(
            {
                "label": label,
                "logical_payload_bytes": elements * 4,
                "median_maximum_rank_seconds": times,
                "dual_speedup_over_fastest_single": fastest_single / times["dual"],
            }
        )
    gradient_times = comparisons[0]["median_maximum_rank_seconds"]
    fastest_single_profile = min(
        ("single_primary", "single_secondary"), key=gradient_times.__getitem__
    )
    if gradient_times["dual"] <= 1.05 * gradient_times[fastest_single_profile]:
        training_profile = "dual"
        selection_reason = "dual_gradient_median_within_5_percent_of_fastest_single"
    else:
        training_profile = fastest_single_profile
        selection_reason = "dual_gradient_median_more_than_5_percent_slower_than_fastest_single"
    payload = {
        "schema_name": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "comparison": "dual_vs_each_single_rail",
        "training_profile": training_profile,
        "training_hcas": NCCL_PROFILES[training_profile],
        "selection_reason": selection_reason,
        "gradient_dual_acceptance_ratio": (
            gradient_times["dual"] / gradient_times[fastest_single_profile]
        ),
        "comparisons": comparisons,
    }
    return payload


def compare_benchmarks(directory: Path) -> dict:
    pair_path = directory / "PAIR.json"
    expected_hosts = None
    if pair_path.is_file() and not pair_path.is_symlink():
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        expected_hosts = pair.get("nodes")
    payload = validate_benchmarks(directory, expected_hosts)
    _atomic_create_json(directory / "NCCL_DUAL_VS_SINGLE.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    node = subparsers.add_parser("audit-node")
    node.add_argument("--out_dir", type=Path, required=True)
    pair = subparsers.add_parser("validate-pair")
    pair.add_argument("--manifest_dir", type=Path, required=True)
    pair.add_argument("--pair", required=True)
    pair.add_argument("--nodes", nargs=2, required=True)
    peer = subparsers.add_parser("verify-peer")
    peer.add_argument("--manifest_dir", type=Path, required=True)
    smoke = subparsers.add_parser("distributed-smoke")
    smoke.add_argument("--out_file", type=Path, required=True)
    smoke.add_argument("--profile", choices=tuple(NCCL_PROFILES), required=True)
    compare = subparsers.add_parser("compare-benchmarks")
    compare.add_argument("--manifest_dir", type=Path, required=True)
    counters = subparsers.add_parser("snapshot-counters")
    counters.add_argument("--out_dir", type=Path, required=True)
    counters.add_argument("--phase", choices=("pre", "post"), required=True)
    counter_compare = subparsers.add_parser("compare-counters")
    counter_compare.add_argument("--manifest_dir", type=Path, required=True)
    counter_compare.add_argument("--nodes", nargs=2, required=True)
    logs = subparsers.add_parser("verify-logs")
    logs.add_argument("--manifest_dir", type=Path, required=True)
    logs.add_argument("--prefix", required=True)
    logs.add_argument("--profile", choices=tuple(NCCL_PROFILES), required=True)
    logs.add_argument("--out_file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit-node":
        audit_node(args.out_dir)
    elif args.command == "validate-pair":
        validate_pair(args.manifest_dir, args.nodes, args.pair)
    elif args.command == "verify-peer":
        verify_peer(args.manifest_dir)
    elif args.command == "distributed-smoke":
        distributed_smoke(args.out_file, args.profile)
    elif args.command == "compare-benchmarks":
        compare_benchmarks(args.manifest_dir)
    elif args.command == "snapshot-counters":
        snapshot_counters(args.out_dir, args.phase)
    elif args.command == "compare-counters":
        compare_counters(args.manifest_dir, args.nodes)
    else:
        verify_nccl_logs(args.manifest_dir, args.prefix, args.profile, args.out_file)


if __name__ == "__main__":
    main()
